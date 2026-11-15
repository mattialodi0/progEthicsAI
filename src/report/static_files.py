"""Access to the report's CSS and JavaScript files in ``report/static``."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

STATIC_DIR = Path(__file__).with_name("static")


@lru_cache(maxsize=None)
def static_text(name: str) -> str:
    """Return the contents of one static file."""
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def script_json(value: Any) -> str:
    """Serialize ``value`` as JSON that is safe inside a ``<script>`` element."""
    return json.dumps(value, ensure_ascii=True).replace("</", "<\/")
