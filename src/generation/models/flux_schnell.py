"""FLUX.1-schnell (too big for most GPUs: runs with CPU offload)."""

from .base import DiffusersGenerator

GENERATOR = DiffusersGenerator(
    name="FLUX.1-schnell",
    version="black-forest-labs/FLUX.1-schnell",
    pipeline="FluxPipeline",
    model_id="black-forest-labs/FLUX.1-schnell",
    dtype="bfloat16",
    cpu_offload=True,
    call_kwargs={
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "height": 768,
        "width": 768,
        "max_sequence_length": 256,
    },
)
