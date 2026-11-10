"""Model adapters: one file per audited text-to-image model.

To add a model, create a file here that defines ``GENERATOR`` (a ``DiffusersGenerator``
for any diffusers pipeline, or an instance of an ``ImageGenerator`` subclass) and
register it in ``MODELS`` below.
"""

from .base import DiffusersGenerator, ImageGenerator
from . import flux2_klein, flux_schnell, pixart_sigma, sd15

MODELS: dict[str, ImageGenerator] = {
    "sd15": sd15.GENERATOR,
    "pixart-sigma": pixart_sigma.GENERATOR,
    "flux-schnell": flux_schnell.GENERATOR,
    "flux2-klein-4b": flux2_klein.GENERATOR,
}

__all__ = ["DiffusersGenerator", "ImageGenerator", "MODELS"]
