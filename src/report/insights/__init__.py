"""Data-driven observations shown in each chapter's "Key observations" box.

Each function reads one chapter payload of the analysis artifact and returns a
list of short HTML sentences. They only describe the sample (largest gaps,
strongest associations, weakest coverage) and always quote the sample size, so
the reader can judge how much weight each observation deserves. Thresholds
are deliberately conservative to avoid highlighting noise from tiny groups.
"""

from .associations import chapter_five, intersectional, stereotypes
from .effects import chapter_eight, chapter_seven, chapter_six
from .measurement import chapter_four, chapter_three, chapter_two
from .summary import conclusive_insights, headline_indicators

__all__ = [
    "chapter_eight", "chapter_five", "chapter_four", "chapter_seven", "chapter_six",
    "chapter_three", "chapter_two", "conclusive_insights", "headline_indicators",
    "intersectional", "stereotypes",
]
