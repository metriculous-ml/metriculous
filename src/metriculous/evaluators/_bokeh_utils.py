from typing import Sequence

from bokeh.models import Div, Title
from bokeh.plotting import Figure

TOOLS = "pan,box_zoom,wheel_zoom,reset"
TOOLBAR_LOCATION = "right"
FONT_SIZE = "8pt"
DARK_BLUE = "#3A5785"
DARK_RED = "#A02444"
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
    p.background_fill_color = "#f5f5f5"
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
