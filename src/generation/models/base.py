"""Base adapter classes shared by every model file in this package."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ImageGenerator(ABC):
    """Adapter between the pipeline and one text-to-image model."""

    #: Model name written in the config and shown in the report.
    name: str
    #: Model version / checkpoint, shown in the report.
    version: str

    @property
    def settings(self) -> dict[str, Any]:
        """Generation parameters shared by every image (recorded for reproducibility)."""
        return {}

    @abstractmethod
    def load(self) -> None:
        """Load the model onto the available device."""

    @abstractmethod
    def generate(self, prompts: list[str], seeds: list[int]) -> list[Any]:
        """Return one PIL image per prompt, each generated from its seed."""


@dataclass
class DiffusersGenerator(ImageGenerator):
    """Generic adapter for any diffusers text-to-image pipeline."""

    name: str
    version: str
    pipeline: str                      # class name in `diffusers`, e.g. "PixArtSigmaPipeline"
    model_id: str                      # Hugging Face repo id
    dtype: str = "float16"             # attribute of `torch`
    call_kwargs: dict[str, Any] = field(default_factory=dict)  # steps, guidance, size...
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    cpu_offload: bool = False          # for models too big for the GPU (e.g. Flux)

    @property
    def settings(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "dtype": self.dtype, **self.call_kwargs}

    def load(self) -> None:
        import diffusers
        import torch

        pipeline_class = getattr(diffusers, self.pipeline)
        self._pipe = pipeline_class.from_pretrained(
            self.model_id, torch_dtype=getattr(torch, self.dtype), **self.load_kwargs
        )
        if self.cpu_offload:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, prompts: list[str], seeds: list[int]) -> list[Any]:
        import torch

        # CPU generators give the same image for a seed on any GPU.
        generators = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
        return self._pipe(prompt=prompts, generator=generators, **self.call_kwargs).images
