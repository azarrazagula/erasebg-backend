import asyncio
import time
import numpy as np
from PIL import Image
from rembg import remove

from services.model_manager import ModelManager
from services.image_re_analyzer import SmallComponentRecovery

class SegmentationPipeline:
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager

    async def run_birefnet(self, image: Image.Image, model_name: str, **kwargs) -> tuple[Image.Image, str, float]:
        """Returns (output_image, model_name, processing_time_seconds)."""
        session = self.mm.get_birefnet_session(model_name)

        matting_params = {"alpha_matting": False}
        matting_params.update(kwargs)

        start = time.perf_counter()
        output_img = await asyncio.to_thread(
            remove,
            image,
            session=session,
            **matting_params
        )
        elapsed = time.perf_counter() - start

        return output_img, model_name, elapsed

    async def run_sam2_complex(self, image: Image.Image) -> tuple[Image.Image, str, float]:
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

        output_img, model_name, birefnet_time = await self.run_birefnet(image_with_sam_mask, "birefnet-general")
        total_elapsed = time.perf_counter() - start
        return output_img, "sam2+birefnet-general", total_elapsed

    async def process(self, image: Image.Image, route: str) -> tuple[Image.Image, str, float]:
        if route == "ROUTE_GRAPHIC":
            output_img, model_name, elapsed = await self.run_birefnet(
                image,
                "birefnet-general",
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=30,
                alpha_matting_erode_size=2
            )
            # Post-process: recover small decorative components
            output_img = await asyncio.to_thread(
                SmallComponentRecovery.recover, image, output_img
            )
            return output_img, model_name, elapsed
        elif route == "ROUTE_PORTRAIT":
            return await self.run_birefnet(image, "birefnet-portrait")
        elif route == "ROUTE_SIMPLE":
            return await self.run_birefnet(image, "birefnet-general")
        elif route == "ROUTE_COMPLEX":
            return await self.run_sam2_complex(image)
        else:
            return await self.run_birefnet(image, "birefnet-general")
