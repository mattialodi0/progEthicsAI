"""FLUX.2-klein 4B (step-distilled: 4 steps, guidance 1.0; runs with CPU offload).

Needs a diffusers release that ships ``Flux2KleinPipeline``.
"""

from .base import DiffusersGenerator

GENERATOR = DiffusersGenerator(
    name="FLUX.2-klein-4B",
    version="black-forest-labs/FLUX.2-klein-4B",
    pipeline="Flux2KleinPipeline",
    model_id="black-forest-labs/FLUX.2-klein-4B",
    dtype="bfloat16",
    cpu_offload=True,
    call_kwargs={
        "height": 784,
        "width": 784,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
    },
)
