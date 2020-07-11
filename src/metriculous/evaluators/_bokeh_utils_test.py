from metriculous.evaluators._bokeh_utils import scatter_plot_circle_size


def test_scatter_plot_circle_size() -> None:
    for num_points, expected_output in [(0, 4.0), (5, 2.25), (10, 0.5)]:
        assert expected_output == scatter_plot_circle_size(
            num_points=num_points,
            biggest=4.0,
            smallest=0.5,
            use_smallest_when_num_points_at_least=10,
        )
