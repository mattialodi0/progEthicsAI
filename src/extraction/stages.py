"""Feature-extraction stages. Each one loads one model and produces a few
fields of the final record; models are imported lazily so only the stages
that run need their dependencies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch / tuple values to plain JSON types."""
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]
    if hasattr(obj, "tolist"):  # numpy arrays / scalars, torch tensors
        return obj.tolist()
    return obj


def best_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Stage(ABC):
    name: str                 # key used on the command line and in the config
    features: list[str]       # analysis-level features it provides
    model: str                # measurement model, reported in the config (empty = none)
    settings: dict[str, Any] = {}

    def load(self) -> None:
        """Load the model (called only if there is something to process)."""

    def unload(self) -> None:
        """Release the model."""

    @abstractmethod
    def process(self, image: Any, record: dict[str, Any]) -> dict[str, Any]:
        """Return the new fields for one record."""


class ImageMetricsStage(Stage):
    """Brightness, saturation and contrast of the image (no model)."""

    name, features, model = "image_metrics", ["image_metrics"], ""

    def process(self, image, record):
        import numpy as np

        gray = np.asarray(image.convert("L"), dtype=np.float32)
        saturation = np.asarray(image.convert("HSV"), dtype=np.float32)[..., 1]
        return {"image_metrics": {
            "brightness": round(float(gray.mean()), 3),
            "saturation": round(float(saturation.mean()), 3),
            "contrast": round(float(gray.std()), 3),
        }}


class FacesStage(Stage):
    """Perceived age, gender, race and emotion of every detected face (DeepFace)."""

    name, features, model = "faces", ["age", "gender", "race", "emotion"], "DeepFace"
    settings = {"actions": ["age", "gender", "race", "emotion"], "detector_backend": "mtcnn",
                "align": True, "enforce_detection": True}

    def load(self) -> None:
        from deepface import DeepFace

        self._deepface = DeepFace

    def process(self, image, record):
        import numpy as np

        try:
            result = self._deepface.analyze(
                img_path=np.asarray(image.convert("RGB"))[..., ::-1],  # DeepFace expects BGR
                actions=self.settings["actions"],
                detector_backend=self.settings["detector_backend"],
                align=self.settings["align"],
                enforce_detection=self.settings["enforce_detection"],
            )
        except ValueError:  # DeepFace raises ValueError when no face is found
            result = []
        return {"facial_feats": to_jsonable(result)}


class FlorenceStage(Stage):
    """Detailed caption and open-vocabulary object labels (Florence-2)."""

    name, features, model = "florence", ["caption", "items_florence"], "Florence-2-base"
    settings = {"model_id": "microsoft/Florence-2-base", "caption_task": "<MORE_DETAILED_CAPTION>",
                "object_task": "<OD>", "num_beams": 3, "max_new_tokens": 512}

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = best_device()
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        model_id = self.settings["model_id"]
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self._dtype, trust_remote_code=True
        ).to(self._device).eval()

    def unload(self) -> None:
        self._model = self._processor = None

    def _run(self, image, task: str):
        import torch

        inputs = self._processor(text=task, images=image, return_tensors="pt").to(
            self._device, self._dtype
        )
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=self.settings["max_new_tokens"],
                num_beams=self.settings["num_beams"],
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        return self._processor.post_process_generation(text, task=task, image_size=image.size)[task]

    def process(self, image, record):
        image = image.convert("RGB")
        return {
            "caption": self._run(image, self.settings["caption_task"]),
            "items_florence": to_jsonable(self._run(image, self.settings["object_task"])),
        }


class ClipStage(Stage):
    """Normalised CLIP embeddings of the image and of its prompt."""

    name, features, model = "clip", ["embedding"], "CLIP ViT-B-32 (OpenAI)"
    settings = {"architecture": "ViT-B-32", "pretrained": "openai"}

    def load(self) -> None:
        import open_clip

        self._device = best_device()
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.settings["architecture"], pretrained=self.settings["pretrained"]
        )
        self._tokenizer = open_clip.get_tokenizer(self.settings["architecture"])
        self._model = self._model.to(self._device).eval()
        self._prompt_cache: dict[str, Any] = {}

    def unload(self) -> None:
        self._model = None

    def process(self, image, record):
        import torch

        prompt = record["prompt"]
        with torch.no_grad():
            if prompt not in self._prompt_cache:  # prompts repeat many times
                text = self._model.encode_text(self._tokenizer(prompt).to(self._device))
                self._prompt_cache[prompt] = (text / text.norm(dim=-1, keepdim=True)).cpu().tolist()
            tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
            embedding = self._model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return {"prompt_embedding": self._prompt_cache[prompt],
                "image_embedding": embedding.cpu().tolist()}


class YoloStage(Stage):
    """COCO object boxes (YOLO11-nano); used for the number of people per image."""

    name, features, model = "yolo", ["items"], "yolo11n"
    settings = {"weights": "yolo11n.pt"}

    def load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.settings["weights"])

    def unload(self) -> None:
        self._model = None

    def process(self, image, record):
        boxes = self._model(image.convert("RGB"), verbose=False)[0].boxes
        return {"items": {
            "xyxy": boxes.xyxy.cpu().tolist(),
            "conf": boxes.conf.cpu().tolist(),
            "cls": boxes.cls.cpu().tolist(),
            "orig_shape": list(boxes.orig_shape),
        }}


STAGES: dict[str, type[Stage]] = {
    stage.name: stage for stage in (ImageMetricsStage, FacesStage, FlorenceStage, ClipStage, YoloStage)
}
