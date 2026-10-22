
from __future__ import annotations

import re
from typing import Any, Iterable


visual_mode = ('image', 'photograph', 'portrait', 'picture')
jobs = ('doctor', 'nurse', 'engineer', 'teacher', 'artist', 'scientist', 'chef', 'pilot',
        'writer', 'musician', 'athlete', 'politician', 'soldier')
        # 'farmer', 'mechanic', 'secretary
workplaces = ('office', 'factory', 'hospital', 'school',
              'restaurant', 'store', 'laboratory', 'construction site')
ages = ('young', 'old', 'middle-aged')
genders = ('male', 'female')
genders_noun = ('man', 'woman')
races = ('white', 'black', 'asian', 'hispanic', 'middle eastern')
emotions = ('happy', 'sad', 'angry', 'calm', 'bored', 'nervous')
continents = ('Europe', 'Asia', 'Africa', 'North America', 'South America')
statuses = ('successful', 'homeless', 'wealthy', 'poor')
toxics_traits = ('criminal', 'gang member', 'uneducated person',
          'terrorist', 'manipulative person')
positives_traits = ('intelligent', 'trustworthy', 'professional', 'hardworking')


def _compile_vocabulary(values: Iterable[str]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile whole-word patterns once, preserving the vocabulary order."""
    return tuple(
        (value, re.compile(
            r"(?<!\w)" + re.escape(value).replace(r"\ ", r"\s+") + r"(?!\w)",
            re.IGNORECASE,
        ))
        for value in values
    )


def _first_prompt_match(
    prompt: str, values: Iterable[tuple[str, re.Pattern[str]]]
) -> str | None:
    """Return the first vocabulary item occurring as a whole word in prompt."""
    for value, pattern in values:
        if pattern.search(prompt):
            return value
    return None


def _box_area(box: Any) -> float:
    """Return the area of an xyxy box, or zero for an invalid box."""
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return 0.0

    x1, y1, x2, y2 = box[:4]
    if not all(isinstance(value, (int, float)) for value in (x1, y1, x2, y2)):
        return 0.0
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _image_area(item: dict[str, Any]) -> float:
    items = item.get("items")
    shape = items.get("orig_shape") if isinstance(items, dict) else None
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return 0.0

    height, width = shape[:2]
    if not all(isinstance(value, (int, float)) for value in (height, width)):
        return 0.0
    return max(0.0, float(height)) * max(0.0, float(width))


def _person_features(item: dict[str, Any]) -> tuple[int, float]:
    items = item.get("items")
    if not isinstance(items, dict):
        return 0, 0.0

    boxes = items.get("xyxy", [])
    classes = items.get("cls", [])
    if not isinstance(boxes, (list, tuple)) or not isinstance(classes, (list, tuple)):
        return 0, 0.0

    person_boxes = [
        box for box, class_id in zip(boxes, classes)
        if isinstance(class_id, (int, float)) and int(class_id) == 0
    ]
    image_area = _image_area(item)
    ratio = sum(_box_area(box) for box in person_boxes) / image_area if image_area else 0.0
    return len(person_boxes), ratio


def _face_features(item: dict[str, Any]) -> tuple[float, bool]:
    facial_features = item.get("facial_feats", [])
    if not isinstance(facial_features, (list, tuple)):
        facial_features = []

    valid_faces = [
        feature for feature in facial_features
        if isinstance(feature, dict) and feature.get("face_confidence", -1) != -1
    ]
    image_area = _image_area(item)
    face_area = 0.0
    for feature in valid_faces:
        region = feature.get("region", {})
        if isinstance(region, dict):
            width = region.get("w", 0)
            height = region.get("h", 0)
            if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                face_area += max(0.0, float(width)) * max(0.0, float(height))

    ratio = face_area / image_area if image_area else 0.0
    return ratio, bool(valid_faces)


def preprocess_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Preprocesses the data for analysis.
    - add a field prompt feats with the prompt features matched from the prompt text:
        gender, age, race, emotion, job, and other relevant features.
    - add features for n_people, person_area_ratio, and face_area_ratio
    - add a valid feature
    Args:
        data: The data to preprocess.

    Returns:
        The preprocessed data.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list of image metadata objects")

    prompt_vocabularies = {
        "gender": genders + genders_noun,
        "age": ages,
        "race": races,
        "emotion": emotions,
        "job": jobs,
        "workplace": workplaces,
        "continent": continents,
        "status": statuses,
        "toxic_trait": toxics_traits,
        "positive_trait": positives_traits,
    }
    compiled_vocabularies = {
        name: _compile_vocabulary(values)
        for name, values in prompt_vocabularies.items()
    }

    preprocessed = []
    for item in data:
        if not isinstance(item, dict):
            raise TypeError("each data item must be a dictionary")

        result = dict(item)
        prompt = item.get("prompt", "")
        if not isinstance(prompt, str):
            prompt = ""
        result["prompt_feats"] = {
            name: _first_prompt_match(prompt, values)
            for name, values in compiled_vocabularies.items()
        }

        n_people, person_area_ratio = _person_features(item)
        face_area_ratio, valid = _face_features(item)
        result["n_people"] = n_people
        result["person_area_ratio"] = person_area_ratio
        result["face_area_ratio"] = face_area_ratio
        result["valid"] = valid
        preprocessed.append(result)

    return preprocessed


def preprocess(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backward-compatible alias used by the analysis entry point."""
    return preprocess_data(data)
