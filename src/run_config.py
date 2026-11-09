"""Run configuration shared by the generation and extraction steps.

One ``config.json`` lives next to the generated data and is updated *while the
run progresses* (status, counters, models, settings, timestamps). The analysis
and report steps later read the same file, so everything the report says about
the audited model (name, settings, extraction models, ...) comes from here.

The file is always rewritten atomically, and keys that a step does not own are
preserved, so hand-edited entries (``prompt_modes``, ``prompt_variables``,
``model_version``, ...) survive re-runs.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PROMPT_MODES = {"0": "control", "1": "single variable", "2": "double variable"}

# Maps the key used in the config to the vocabulary name in prompt_generator.
_VARIABLE_SOURCES = {
    "jobs": "jobs",
    "workplaces": "workplaces",
    "ages": "ages",
    "genders": "genders",
    "genders_noun": "genders_noun",
    "races": "races",
    "emotions": "emotions",
    "continents": "continents",
    "statuses": "statuses",
    "toxic_traits": "toxics",
    "positive_traits": "positives",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def default_prompt_variables() -> dict[str, list[str]]:
    """Prompt vocabularies, taken from the prompt generator so they never drift."""
    from .generation import prompt_generator as pg

    return {key: list(getattr(pg, attr)) for key, attr in _VARIABLE_SOURCES.items()}


def environment_info() -> dict[str, Any]:
    """Versions and hardware of the machine that ran the heavy steps."""
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("torch", "diffusers", "transformers", "deepface", "ultralytics", "open_clip"):
        try:
            imported = __import__(module)
            info[module] = getattr(imported, "__version__", "unknown")
        except Exception:  # not installed in this step's environment
            continue
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


class RunConfig:
    """Load, update and atomically persist the run's ``config.json``."""

    def __init__(self, path: str | Path, save_interval: float = 5.0):
        self.path = Path(path)
        self.save_interval = save_interval
        self._last_save = 0.0
        self.data: dict[str, Any] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self._fill_defaults()

    def _fill_defaults(self) -> None:
        self.data.setdefault("prompt_modes", dict(PROMPT_MODES))
        self.data.setdefault("prompt_variables", default_prompt_variables())
        self.data.setdefault("extracted_feats", [])
        self.data.setdefault("extraction_models", [])

    # -- access ---------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def section(self, name: str) -> dict[str, Any]:
        """Return (creating if needed) a mutable top-level section."""
        value = self.data.get(name)
        if not isinstance(value, dict):
            value = self.data[name] = {}
        return value

    def update(self, **values: Any) -> None:
        self.data.update(values)
        self.save()

    def update_section(self, name: str, force: bool = True, **values: Any) -> None:
        self.section(name).update(values)
        self.save(force=force)

    # -- persistence ----------------------------------------------------
    def save(self, force: bool = True) -> None:
        """Write the file. ``force=False`` is throttled, for per-item progress."""
        now = time.monotonic()
        if not force and now - self._last_save < self.save_interval:
            return
        self._last_save = now
        self.data["updated_at"] = now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.path)
