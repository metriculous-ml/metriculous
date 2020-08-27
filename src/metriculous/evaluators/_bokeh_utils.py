from typing import Sequence, Tuple, cast

import numpy as np
from bokeh.embed import file_html
from bokeh.models import Div, Title
from bokeh.plotting import Figure
from bokeh.resources import CDN

TOOLS = "pan,box_zoom,wheel_zoom,reset"
TOOLBAR_LOCATION = "right"
FONT_SIZE = "8pt"
DARK_BLUE = "#3A5785"
DARK_RED = "#A02444"
BACKGROUND_COLOR = "#F5F5F5"
HISTOGRAM_ALPHA = 0.5
SCATTER_CIRCLES_FILL_ALPHA = 0.5
SCATTER_CIRCLES_LINE_ALPHA = 0.9


def title_div(title_rows: Sequence[str]) -> Div:
    return Div(
        text="".join(
            f"""
                <div style="
                    color: rgba(0,0,0,0.7);
                    text-align: center;
                    font-size: {FONT_SIZE};
                    font-weight: bold;
                ">
                {title_row}
                </div>
                """
            for title_row in title_rows
        ),
        sizing_mode="stretch_width",
        align="center",
    )


def add_title_rows(p: Figure, title_rows: Sequence[str]) -> None:
    for title_row in reversed(title_rows):
        p.add_layout(
            Title(text=title_row, text_font_size=FONT_SIZE, align="center"),
            place="above",
        )


def apply_default_style(p: Figure) -> None:
    p.background_fill_color = BACKGROUND_COLOR
    p.grid.grid_line_color = "white"

    p.toolbar.logo = None

    p.xaxis.axis_label_text_font_size = FONT_SIZE
    p.yaxis.axis_label_text_font_size = FONT_SIZE

    p.axis.axis_line_color = None

    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks

    p.axis.major_label_standoff = 0

    if p.legend:
        p.legend.label_text_font_size = FONT_SIZE
        p.legend.background_fill_alpha = 0.85


def scatter_plot_circle_size(
    num_points: int,
    biggest: float = 3.0,
    smallest: float = 1.0,
    use_smallest_when_num_points_at_least: float = 5000,
) -> float:
    assert biggest >= smallest
    slope = (biggest - smallest) / use_smallest_when_num_points_at_least
    return max(smallest, biggest - slope * num_points)


def check_that_all_figures_can_be_rendered(figures: Sequence[Figure]) -> None:
    """ Generates HTML for each figure.

    In some cases this reveals issues that might not be noticed if we just instantiated the figures
    without showing them or generating their HTML representations.
    """
    for f in figures:
        html = file_html(f, resources=CDN)
        assert isinstance(html, str)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    assert hex_color.startswith("#")
    assert len(hex_color) == 7, hex_color
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return r, g, b


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def color_palette(start_hex_color: str, end_hex_color: str, n: int) -> Sequence[str]:
    assert n >= 2, n
    start_rgb = np.array(hex_to_rgb(start_hex_color))
    end_rgb = np.array(hex_to_rgb(end_hex_color))
    step_rgb = (end_rgb - start_rgb) / (n - 1)
    rgb_palette: Sequence[Tuple[int, int, int]] = [
        cast(
            Tuple[int, int, int], tuple(np.rint(start_rgb + i * step_rgb).astype("int"))
        )
        for i in range(n)
    ]
    return [rgb_to_hex(rgb) for rgb in rgb_palette]
