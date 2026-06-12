import asyncio
import time
import numpy as np
from PIL import Image
from rembg import remove        # BiRefNet inference library

from models.loader import ModelLoader
from services.resolution_manager import ResolutionManager       # Resize logic
from services.image_re_analyzer import SmallComponentRecovery   # Post-processing


class SegmentationPipeline:
    """
    Executes the BiRefNet model inference.

    Improvements over original:
      1. Resolution Capping: Uses ResolutionManager to downscale the image
         *before* inference (e.g., 4032x3024 -> 1024x768).
      2. Mask Upscaling: Extracts the alpha mask from the low-res inference
         output and upscales it back to the original image dimensions using
         edge-preserving filters (cv2.bilateralFilter).
      3. alpha_matting is DISABLED on ROUTE_GRAPHIC (saves 5-15s).
      4. SAM2 is preserved but rarely used (controlled by router).
    """

    def __init__(self, model_loader: ModelLoader):
        self.mm = model_loader  # ModelLoader singleton reference

    async def run_birefnet(
        self,
        image: Image.Image,
        model_name: str,
        **kwargs
    ) -> tuple[Image.Image, str, float]:
        """
        Runs inference on the provided image using BiRefNet.
        Returns: (low_res_rgba_output, model_name, inference_time_seconds)
        """
        # ONNX session எடு (cache-ல இருந்தா reuse, இல்லன்னா load)
        session = self.mm.get_session(model_name)

        # alpha_matting=False default — GrabCut CPU penalty avoid
        matting_params = {"alpha_matting": False}
        matting_params.update(kwargs)   # Caller-ல override பண்ணினா allow

        start = time.perf_counter()
        # remove() blocking call — asyncio thread pool-ல run பண்ணு (event loop block ஆகாது)
        output_img = await asyncio.to_thread(
            remove,
            image,
            session=session,
            **matting_params
        )
        elapsed = time.perf_counter() - start

        # Apple Silicon MPS memory free பண்ணு (torch.mps.empty_cache())
        self.mm.clear_mps_cache()

        return output_img, model_name, elapsed

    async def run_sam2_complex(self, image: Image.Image) -> tuple[Image.Image, str, float]:
        """
        Runs SAM2 + BiRefNet-general. Very slow on MPS (20s+).
        SAM2 center-point prompt மூலம் foreground mask எடுத்து BiRefNet-க்கு feed பண்ணும்.
        """
        predictor = await self.mm.get_sam2_predictor()
        cv_img = np.array(image.convert("RGB"))  # PIL → numpy BGR

        def _predict():
            predictor.set_image(cv_img)
            h, w = cv_img.shape[:2]
            point_coords = np.array([[w // 2, h // 2]])  # Image center point
            point_labels = np.array([1])                  # 1 = foreground
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False    # Single mask மட்டும்
            )
            return masks[0]

        start = time.perf_counter()
        mask = await asyncio.to_thread(_predict)

        # SAM2 mask → alpha channel-ஆ convert
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        image_with_sam_mask = image.copy().convert("RGBA")
        image_with_sam_mask.putalpha(mask_img)  # SAM2 mask-ஐ alpha-ஆ set

        # SAM2 result-ஐ BiRefNet-general-க்கு refine பண்ண pass பண்ணு
        output_img, model_name, _ = await self.run_birefnet(image_with_sam_mask, "birefnet-general")
        total_elapsed = time.perf_counter() - start
        return output_img, "sam2+birefnet-general", total_elapsed

    async def process(
        self, original_image: Image.Image, route: str
    ) -> tuple[Image.Image, str, float]:
        """
        Full processing pipeline:
          1. Resize to inference resolution
          2. Run Inference
          3. Upscale mask back to original resolution
          4. Small Component Recovery (if ROUTE_GRAPHIC)
        """
        import time
        t_start = time.perf_counter()

        # Step 1: Inference-க்கு ஏத்த size-க்கு resize (4K → 1024px)
        # original_image full res, infer_image small res
        infer_image, orig_size = ResolutionManager.resize_for_inference(
            original_image, route
        )
        t_resize = time.perf_counter()
        print(f"[Timing] Resize Time: {t_resize - t_start:.4f}s")

        # Step 2: Route-க்கு ஏத்த model run பண்ணு
        if route == "ROUTE_GRAPHIC":
            # alpha_matting=False explicitly — graphic-க்கு GrabCut பண்றது wrong
            infer_out, model_name, elapsed = await self.run_birefnet(
                infer_image, "birefnet-general", alpha_matting=False
            )
        elif route == "ROUTE_PORTRAIT":
            # Portrait model — hair/skin edge detail better
            infer_out, model_name, elapsed = await self.run_birefnet(
                infer_image, "birefnet-portrait"
            )
        elif route == "ROUTE_COMPLEX":
            # SAM2 + BiRefNet combo (rarely triggered)
            infer_out, model_name, elapsed = await self.run_sam2_complex(infer_image)
        else:
            # ROUTE_SIMPLE — general model, animals, objects, nature
            infer_out, model_name, elapsed = await self.run_birefnet(
                infer_image, "birefnet-general"
            )
        t_infer = time.perf_counter()
        print(f"[Timing] Inference Time: {t_infer - t_resize:.4f}s")

        # Step 3: Low-res mask-ஐ original resolution-க்கு upscale + composite
        # infer_out = small RGBA, original_image = full res → final = full res RGBA
        final_out = await asyncio.to_thread(
            ResolutionManager.upscale_mask_to_original,
            infer_out,
            original_image,
            orig_size
        )
        t_upscale = time.perf_counter()
        print(f"[Timing] Mask Upscale Time: {t_upscale - t_infer:.4f}s")

        # Step 4: Graphic route மட்டும் — missed small components recover பண்ணு
        # (logos, dots, thin lines BiRefNet miss பண்ணியிருக்கும்)
        if route == "ROUTE_GRAPHIC":
            final_out = await asyncio.to_thread(
                SmallComponentRecovery.recover, original_image, final_out
            )
            t_recovery = time.perf_counter()
            print(f"[Timing] Recovery Time: {t_recovery - t_upscale:.4f}s")

        return final_out, model_name, time.perf_counter() - t_start
