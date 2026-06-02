import asyncio
import time
import numpy as np
from PIL import Image
from rembg import remove

from services.model_manager import ModelManager
from services.resolution_manager import ResolutionManager
from services.image_re_analyzer import SmallComponentRecovery


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

    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager

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
        session = self.mm.get_birefnet_session(model_name)

        matting_params = {"alpha_matting": False}
        matting_params.update(kwargs)

        start = time.perf_counter()
        # remove() blocking call runs in thread pool
        output_img = await asyncio.to_thread(
            remove,
            image,
            session=session,
            **matting_params
        )
        elapsed = time.perf_counter() - start

        # Free MPS memory after inference
        self.mm._clear_mps_cache()

        return output_img, model_name, elapsed

    async def run_sam2_complex(self, image: Image.Image) -> tuple[Image.Image, str, float]:
        """
        Runs SAM2 + BiRefNet-general. Very slow on MPS (20s+).
        """
        predictor = await self.mm.get_sam2_predictor()
        cv_img = np.array(image.convert("RGB"))

        def _predict():
            predictor.set_image(cv_img)
            h, w = cv_img.shape[:2]
            point_coords = np.array([[w // 2, h // 2]])
            point_labels = np.array([1])
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
            return masks[0]

        start = time.perf_counter()
        mask = await asyncio.to_thread(_predict)

        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        image_with_sam_mask = image.copy().convert("RGBA")
        image_with_sam_mask.putalpha(mask_img)

        # Run birefnet-general on the SAM2 result
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

        # 1. Resize for inference
        infer_image, orig_size = ResolutionManager.resize_for_inference(
            original_image, route
        )
        t_resize = time.perf_counter()
        print(f"[Timing] Resize Time: {t_resize - t_start:.4f}s")

        # 2. Run Inference
        if route == "ROUTE_GRAPHIC":
            # BUGFIX: alpha_matting=False for graphics. 
            # Original had True, causing full-res GrabCut on CPU (50s penalty).
            infer_out, model_name, elapsed = await self.run_birefnet(
                infer_image, "birefnet-general", alpha_matting=False
            )
        elif route == "ROUTE_PORTRAIT":
            infer_out, model_name, elapsed = await self.run_birefnet(
                infer_image, "birefnet-portrait"
            )
        elif route == "ROUTE_COMPLEX":
            infer_out, model_name, elapsed = await self.run_sam2_complex(infer_image)
        else:
            # ROUTE_SIMPLE
            infer_out, model_name, elapsed = await self.run_birefnet(
                infer_image, "birefnet-general"
            )
        t_infer = time.perf_counter()
        print(f"[Timing] Inference Time: {t_infer - t_resize:.4f}s")

        # 3. Upscale mask and composite with original image
        final_out = await asyncio.to_thread(
            ResolutionManager.upscale_mask_to_original,
            infer_out,
            original_image,
            orig_size
        )
        t_upscale = time.perf_counter()
        print(f"[Timing] Mask Upscale Time: {t_upscale - t_infer:.4f}s")

        # 4. Small Component Recovery (runs on final upscaled mask)
        if route == "ROUTE_GRAPHIC":
            final_out = await asyncio.to_thread(
                SmallComponentRecovery.recover, original_image, final_out
            )
            t_recovery = time.perf_counter()
            print(f"[Timing] Recovery Time: {t_recovery - t_upscale:.4f}s")

        return final_out, model_name, time.perf_counter() - t_start
