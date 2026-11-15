"""One renderer per report chapter; each returns the chapter's HTML."""

from .intro import render_reading_guide
from .ch1_summary import render_chapter_one
from .ch2_coverage import render_chapter_two
from .ch3_representation import render_chapter_three
from .ch4_adherence import render_chapter_four
from .ch5_associations import render_chapter_five
from .ch5_objects import render_stereotypical_associations
from .ch5_intersectional import render_intersectional_analysis
from .ch6_counterfactual import render_chapter_six
from .ch7_visual import render_chapter_seven
from .ch8_captions import render_chapter_eight
from .ch9_synthesis import render_chapter_nine
from .ch10_methodology import render_chapter_ten

__all__ = [
    "render_chapter_eight", "render_chapter_five", "render_chapter_four",
    "render_chapter_nine", "render_chapter_one", "render_chapter_seven",
    "render_chapter_six", "render_chapter_ten", "render_chapter_three",
    "render_chapter_two", "render_intersectional_analysis", "render_reading_guide",
    "render_stereotypical_associations",
]
