"""PixArt-Sigma XL 1024."""

from .base import DiffusersGenerator

GENERATOR = DiffusersGenerator(
    name="PixArtSigma-XL-2-1024-MS",
    version="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    pipeline="PixArtSigmaPipeline",
    model_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    call_kwargs={"num_inference_steps": 20, "guidance_scale": 4.5},
)
