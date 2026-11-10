"""Stable Diffusion 1.5."""

from .base import DiffusersGenerator

GENERATOR = DiffusersGenerator(
    name="StableDiffusion-1.5",
    version="runwayml/stable-diffusion-v1-5",
    pipeline="StableDiffusionPipeline",
    model_id="runwayml/stable-diffusion-v1-5",
    call_kwargs={"num_inference_steps": 30, "guidance_scale": 7.5},
)
