"""test_simple_plotter.py.

Last Updated: February 13, 2025

For the scattermapbox deprecation warning, see https://github.com/plotly/plotly.py/issues/4997.
In Plotly 6.0.0, plotly.pio also uses a deprecated kaleido method. At some point an update of
Plotly should eliminate these warnings.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly.express as px
import pytest
from plotly.graph_objects import Figure
from pydantic import ValidationError

from lexos.exceptions import LexosException
from lexos.rolling_windows.plotters.plotly_plotter import PlotlyPlotter

# Fixtures

def kaleido_available():
    """Check if kaleido is available for image export."""
    try:
        import kaleido
        return True
    except ImportError:
        return False


@pytest.fixture
def basic_plotter():
    """Creates a PlotlyPlotter instance with default settings.

    Returns:
        PlotlyPlotter: A plotter instance with default configuration
    """
    return PlotlyPlotter()


@pytest.fixture
def valid_milestone_labels():
    """Creates valid milestone labels for testing.

    Returns:
        dict: Dictionary of milestone labels and positions
    """
    return {"Start": 0, "Middle": 50, "End": 100}


@pytest.fixture
def basic_line_plot():
    """Create a PlotlyPlotter instance with a basic figure.

    Returns:
        PlotlyPlotter: Configured plotter instance with basic figure
    """
    plotter = PlotlyPlotter()
    df = pd.DataFrame({"test": [1, 2, 3]})
    plotter.fig = px.line(df)
    return plotter


@pytest.fixture
def milestone_plotter():
    """Create a PlotlyPlotter instance with milestone settings.

    Returns:
        PlotlyPlotter: Plotter instance configured with milestone settings
    """
    return PlotlyPlotter(
        milestone_label_rotation=45,
        milestone_label_style={"size": 12, "family": "Arial", "color": "red"},
    )


@pytest.fixture
def empty_plotter():
    """Creates a PlotlyPlotter instance without a figure.

    Returns:
        PlotlyPlotter: Plotter instance without figure
    """
    return PlotlyPlotter()


# Tests


def test_default_initialization(basic_plotter):
    """Tests initialization with default values."""
    assert basic_plotter.id == "rw_plotly_plotter"
    assert basic_plotter.width == 700
    assert basic_plotter.height == 450
    assert basic_plotter.title == "Rolling Windows Plot"
    assert basic_plotter.xlabel == "Token Count"
    assert basic_plotter.ylabel == "Average Frequency"
    assert basic_plotter.line_color == px.colors.qualitative.Plotly
    assert basic_plotter.showlegend is True
    assert basic_plotter.titlepad is None
    assert basic_plotter.fig is None


def test_custom_initialization():
    """Tests initialization with custom values."""
    custom_plotter = PlotlyPlotter(
        width=1000, height=800, title="Custom Title", line_color="red"
    )
    assert custom_plotter.width == 1000
    assert custom_plotter.height == 800
    assert custom_plotter.title == "Custom Title"
    assert custom_plotter.line_color == "red"


def test_milestone_default_values(basic_plotter):
    """Tests default milestone-related attributes."""
    assert basic_plotter.show_milestones is False
    assert basic_plotter.show_milestone_labels is False
    assert basic_plotter.milestone_labels is None
    assert basic_plotter.milestone_label_rotation == 0.0
    assert basic_plotter.milestone_marker_style == {"width": 1, "color": "teal"}


def test_milestone_label_style(basic_plotter):
    """Tests default milestone label style settings."""
    expected_style = {
        "size": 10.0,
        "family": "Open Sans, verdana, arial, sans-serif",
        "color": "teal",
    }
    assert basic_plotter.milestone_label_style == expected_style


def test_invalid_rotation():
    """Tests validation of milestone label rotation."""
    with pytest.raises(LexosException) as exc_info:
        PlotlyPlotter(milestone_label_rotation=91)
    assert "maximum of 90 degrees" in str(exc_info.value)


def test_milestone_validation():
    """Tests validation when milestones are enabled but labels are missing."""
    with pytest.raises(LexosException) as exc_info:
        PlotlyPlotter(show_milestones=True)
    assert "milestone_labels" in str(exc_info.value)


def test_valid_milestone_configuration():
    """Tests valid milestone configuration."""
    milestone_labels = {"Chapter 1": 100, "Chapter 2": 200}
    plotter = PlotlyPlotter(
        show_milestones=True,
        show_milestone_labels=True,
        milestone_labels=milestone_labels,
    )
    assert plotter.milestone_labels == milestone_labels


def test_valid_rotation_values():
    """Tests valid milestone label rotation values."""
    valid_rotations = [0, 45, 90]
    for rotation in valid_rotations:
        plotter = PlotlyPlotter(milestone_label_rotation=rotation)
        assert plotter.milestone_label_rotation == rotation


def test_invalid_rotation_value():
    """Tests that invalid rotation values raise LexosException."""
    invalid_rotations = [91, 180, 360]
    for rotation in invalid_rotations:
        with pytest.raises(LexosException) as exc_info:
            PlotlyPlotter(milestone_label_rotation=rotation)
        assert "maximum of 90 degrees" in str(exc_info.value)


def test_rotation_edge_cases():
    """Tests rotation edge cases."""
    # Test exactly 90 degrees
    plotter = PlotlyPlotter(milestone_label_rotation=90)
    assert plotter.milestone_label_rotation == 90

    # Test negative values
    plotter = PlotlyPlotter(milestone_label_rotation=-45)
    assert plotter.milestone_label_rotation == -45


@pytest.mark.parametrize(
    "rotation",
    [
        0.0,  # Zero rotation
        45.5,  # Decimal value
        90.0,  # Maximum allowed value
        -90,  # Negative value
    ],
)
def test_rotation_parameterized(rotation):
    """Tests various rotation values using parameterization.

    Args:
        rotation (float): The rotation value to test
    """
    plotter = PlotlyPlotter(milestone_label_rotation=rotation)
    assert plotter.milestone_label_rotation == rotation


def test_validate_edge_cases_no_milestones():
    """Tests validation when milestones are disabled."""
    plotter = PlotlyPlotter()
    plotter._validate_edge_cases()  # Should not raise exception


def test_validate_edge_cases_with_valid_milestones(valid_milestone_labels):
    """Tests validation with valid milestone configuration.

    Args:
        valid_milestone_labels: Fixture providing valid milestone labels
    """
    plotter = PlotlyPlotter(
        show_milestones=True, milestone_labels=valid_milestone_labels
    )
    plotter._validate_edge_cases()  # Should not raise exception


def test_validate_edge_cases_missing_labels():
    """Tests validation when milestone labels are missing but required."""
    with pytest.raises(LexosException) as exc_info:
        plotter = PlotlyPlotter(show_milestones=True)
        plotter._validate_edge_cases()
    assert "milestone_labels" in str(exc_info.value)


def test_validate_edge_cases_invalid_labels():
    """Tests validation with invalid milestone labels."""
    invalid_cases = [
        {},  # Empty dict
        {"Label": "invalid"},  # Invalid value type
        {1: 100},  # Invalid key type
    ]

    for invalid_labels in invalid_cases:
        with pytest.raises((LexosException, ValidationError)) as exc_info:
            plotter = PlotlyPlotter(
                show_milestones=True, milestone_labels=invalid_labels
            )
            plotter._validate_edge_cases()
        assert "milestone_labels" in str(exc_info.value)

def test_validate_edge_cases_invalid_milestone_values():
    """Test validation with milestone labels that cause MilestonesModel ValidationError (line 155 coverage)."""
    from unittest.mock import patch
    
    # Create a plotter instance first with valid data
    plotter = PlotlyPlotter(show_milestones=True, milestone_labels={"test": 1})
    
    # Set milestone_labels to non-empty value to avoid line 150-153
    plotter.milestone_labels = {"some_label": 123}
    
    # Mock MilestonesModel to raise ValidationError using the simplest working approach
    with patch('lexos.rolling_windows.plotters.plotly_plotter.MilestonesModel') as mock_model:
        # Create ValidationError by trying to validate invalid data with a real Pydantic model
        try:
            from pydantic import BaseModel
            class TestModel(BaseModel):
                required_field: str
            TestModel()  # This will raise ValidationError for missing required field
        except ValidationError as ve:
            mock_model.side_effect = ve
        
        with pytest.raises(LexosException) as exc_info:
            plotter._validate_edge_cases()
        
        # Verify it's the specific error from line 155 (the else clause)
        assert "require a value for `milestone_labels`" in str(exc_info.value)
        assert "list of dicts" in str(exc_info.value)

def test_validate_edge_cases_show_labels_only(valid_milestone_labels):
    """Tests validation when only milestone labels are shown.

    Args:
        valid_milestone_labels: Fixture providing valid milestone labels
    """
    plotter = PlotlyPlotter(
        show_milestone_labels=True, milestone_labels=valid_milestone_labels
    )
    plotter._validate_edge_cases()  # Should not raise exception


def test_validate_edge_cases_both_options(valid_milestone_labels):
    """Tests validation with both milestone options enabled.

    Args:
        valid_milestone_labels: Fixture providing valid milestone labels
    """
    plotter = PlotlyPlotter(
        show_milestones=True,
        show_milestone_labels=True,
        milestone_labels=valid_milestone_labels,
    )
    plotter._validate_edge_cases()  # Should not raise exception


def test_init_with_custom_values():
    """Tests initialization with custom parameters."""
    custom_values = {
        "width": 800,
        "height": 600,
        "title": "Custom Title",
        "line_color": "blue",
    }
    plotter = PlotlyPlotter(**custom_values)

    for key, value in custom_values.items():
        assert getattr(plotter, key) == value


def test_init_with_invalid_milestone_config():
    """Tests initialization with invalid milestone configuration."""
    with pytest.raises(LexosException) as exc_info:
        PlotlyPlotter(show_milestones=True, milestone_labels=None)
    assert "milestone_labels" in str(exc_info.value)


def test_init_with_valid_milestone_config(valid_milestone_labels):
    """Tests initialization with valid milestone configuration.

    Args:
        valid_milestone_labels: Fixture providing valid milestone labels
    """
    plotter = PlotlyPlotter(
        show_milestones=True, milestone_labels=valid_milestone_labels
    )
    assert plotter.milestone_labels == valid_milestone_labels


def test_init_attribute_override():
    """Tests that initialization properly overrides default attributes."""
    default_plotter = PlotlyPlotter()
    custom_plotter = PlotlyPlotter(width=1000)

    assert default_plotter.width == 700
    assert custom_plotter.width == 1000


def test_init_invalid_attribute():
    """Tests initialization with invalid attribute."""
    plotter = PlotlyPlotter(invalid_attr="test")
    assert hasattr(plotter, "invalid_attr")
    assert plotter.invalid_attr == "test"


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"width": 800}, 800),
        ({"height": 600}, 600),
        ({"title": "Test"}, "Test"),
        ({"showlegend": False}, False),
    ],
)
def test_init_parameters(test_input, expected):
    """Tests initialization with different parameter combinations.

    Args:
        test_input: Dictionary of input parameters
        expected: Expected value for the parameter
    """
    plotter = PlotlyPlotter(**test_input)
    for key, value in test_input.items():
        assert getattr(plotter, key) == expected


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing.

    Returns:
        pd.DataFrame: Sample DataFrame with test data
    """
    return pd.DataFrame({"term1": [1, 2, 3, 4, 5], "term2": [2, 4, 6, 8, 10]})


def test_call_basic_plot(basic_plotter, sample_df):
    """Test basic plot creation with default settings."""
    basic_plotter(df=sample_df, show_plot=False)
    assert basic_plotter.fig is not None
    assert isinstance(basic_plotter.fig, Figure)


def test_call_custom_dimensions(basic_plotter, sample_df):
    """Test plot creation with custom dimensions."""
    basic_plotter(df=sample_df, width=800, height=600, show_plot=False)
    assert basic_plotter.fig.layout.width == 800
    assert basic_plotter.fig.layout.height == 600


def test_call_with_custom_title(basic_plotter, sample_df):
    """Test plot creation with custom title."""
    custom_title = "Custom Plot Title"
    basic_plotter(df=sample_df, title=custom_title, show_plot=False)
    assert basic_plotter.fig.layout.title.text == custom_title


def test_call_with_milestones(basic_plotter, sample_df):
    """Test plot creation with milestones."""
    milestone_labels = {"Start": 0, "Middle": 2, "End": 4}
    basic_plotter(
        df=sample_df,
        show_milestones=True,
        milestone_labels=milestone_labels,
        show_plot=False,
    )
    assert len(basic_plotter.fig.layout.shapes) == 3


def test_call_with_milestone_labels(basic_plotter, sample_df):
    """Test plot creation with milestone labels."""
    milestone_labels = {"Start": 0, "Middle": 2, "End": 4}
    basic_plotter(
        df=sample_df,
        show_milestones=True,
        milestone_labels=milestone_labels,
        show_milestone_labels=True,
        show_plot=False,
    )
    assert len(basic_plotter.fig.layout.shapes) == 3
    assert len(basic_plotter.milestone_labels) == 3
    assert len(basic_plotter.fig.layout.annotations) == 3


def test_call_invalid_df():
    """Test plot creation with invalid DataFrame."""
    plotter = PlotlyPlotter()
    with pytest.raises(Exception):
        plotter(df=None)


def test_call_custom_line_color(basic_plotter, sample_df):
    """Test plot creation with custom line color."""
    basic_plotter(df=sample_df, line_color="red", show_plot=False)
    # basic_plotter(df=sample_df, show_plot=False)
    assert basic_plotter.line_color == "red"


@pytest.mark.parametrize("show_legend", [True, False])
def test_call_legend_visibility(basic_plotter, sample_df, show_legend):
    """Test legend visibility settings.

    Args:
        show_legend (bool): Legend visibility setting to test
    """
    basic_plotter(df=sample_df, showlegend=show_legend, show_plot=False)
    assert basic_plotter.fig.layout.showlegend == show_legend


def test_call_with_kwargs(basic_plotter, sample_df):
    """Test plot creation with additional kwargs."""
    basic_plotter(df=sample_df, show_plot=False, template="plotly_dark")
    # There is no way to get the name of the template, but this approimates it
    # and ensures that we have the right template set.
    assert basic_plotter.fig.layout.template.layout.mapbox.style == "dark"


def test_plot_milestone_label_basic(basic_line_plot):
    """Test basic milestone label plotting."""
    basic_line_plot._plot_milestone_label("Test Label", 100)

    annotations = basic_line_plot.fig.layout.annotations
    assert len(annotations) == 1
    assert annotations[0].text == "Test Label"
    assert annotations[0].x == 100


def test_plot_milestone_label_position(basic_line_plot):
    """Test milestone label positioning."""
    basic_line_plot._plot_milestone_label("Test", 50)

    annotation = basic_line_plot.fig.layout.annotations[0]
    assert annotation.y == 1
    assert annotation.xanchor == "left"
    assert annotation.yanchor == "bottom"
    assert annotation.xshift == -10
    assert annotation.yref == "paper"
    assert annotation.showarrow is False


def test_plot_milestone_label_rotation(milestone_plotter):
    """Test milestone label rotation."""
    milestone_plotter.fig = px.line(pd.DataFrame({"test": [1, 2, 3]}))
    milestone_plotter._plot_milestone_label("Test", 0)

    annotation = milestone_plotter.fig.layout.annotations[0]
    assert annotation.textangle == -45  # Negative because of the rotation direction


def test_plot_milestone_label_style(milestone_plotter):
    """Test milestone label styling."""
    milestone_plotter.fig = px.line(pd.DataFrame({"test": [1, 2, 3]}))
    milestone_plotter._plot_milestone_label("Test", 0)

    annotation = milestone_plotter.fig.layout.annotations[0]
    assert annotation.font.size == 12
    assert annotation.font.family == "Arial"
    assert annotation.font.color == "red"


def test_plot_milestone_label_multiple(basic_line_plot):
    """Test plotting multiple milestone labels."""
    labels_and_positions = [("Label 1", 10), ("Label 2", 20), ("Label 3", 30)]

    for label, pos in labels_and_positions:
        basic_line_plot._plot_milestone_label(label, pos)

    annotations = basic_line_plot.fig.layout.annotations
    assert len(annotations) == 3

    for i, (label, pos) in enumerate(labels_and_positions):
        assert annotations[i].text == label
        assert annotations[i].x == pos


@pytest.mark.parametrize(
    "label,x_pos",
    [
        ("Short", 0),
        ("Very Long Label", 100),
        ("Special $#@! Chars", 50),
        ("", 25),  # Empty label
    ],
)
def test_plot_milestone_label_different_inputs(basic_line_plot, label, x_pos):
    """Test milestone label plotting with various inputs.

    Args:
        label (str): Label text to test
        x_pos (int): X-axis position to test
    """
    basic_line_plot._plot_milestone_label(label, x_pos)

    annotation = basic_line_plot.fig.layout.annotations[0]
    assert annotation.text == label
    assert annotation.x == x_pos


def test_plot_milestone_marker_basic(basic_line_plot):
    """Tests basic milestone marker plotting."""
    basic_line_plot._plot_milestone_marker(x=50, df_val_min=0, df_val_max=100)

    shapes = basic_line_plot.fig.layout.shapes
    assert len(shapes) == 1
    assert shapes[0].type == "line"
    assert shapes[0].x0 == shapes[0].x1 == 50
    assert shapes[0].y0 == 0
    assert shapes[0].y1 == 100


def test_plot_milestone_marker_style(basic_line_plot):
    """Tests milestone marker styling."""
    custom_style = {"width": 2, "color": "red", "dash": "dash"}
    basic_line_plot.milestone_marker_style = custom_style
    basic_line_plot._plot_milestone_marker(x=50, df_val_min=0, df_val_max=100)

    shape = basic_line_plot.fig.layout.shapes[0]
    assert shape.line.width == 2
    assert shape.line.color == "red"
    assert shape.line.dash == "dash"


def test_plot_milestone_marker_references(basic_line_plot):
    """Tests milestone marker axis references."""
    basic_line_plot._plot_milestone_marker(x=50, df_val_min=0, df_val_max=100)

    shape = basic_line_plot.fig.layout.shapes[0]
    assert shape.xref == "x"
    assert shape.yref == "y"


def test_plot_milestone_marker_multiple(basic_line_plot):
    """Tests plotting multiple milestone markers."""
    markers = [(10, 0, 100), (20, 0, 100), (30, 0, 100)]

    for x, min_val, max_val in markers:
        basic_line_plot._plot_milestone_marker(x, min_val, max_val)

    shapes = basic_line_plot.fig.layout.shapes
    assert len(shapes) == 3

    for i, (x, min_val, max_val) in enumerate(markers):
        assert shapes[i].x0 == shapes[i].x1 == x
        assert shapes[i].y0 == min_val
        assert shapes[i].y1 == max_val


@pytest.mark.parametrize(
    "x,min_val,max_val", [(0, 0, 10), (100, -10, 0), (50, -100, 100), (25, 0.5, 1.5)]
)
def test_plot_milestone_marker_different_values(basic_line_plot, x, min_val, max_val):
    """Tests milestone marker plotting with various input values.

    Args:
        x (int): X-axis position
        min_val (float): Minimum y-value
        max_val (float): Maximum y-value
    """
    basic_line_plot._plot_milestone_marker(x, min_val, max_val)

    shape = basic_line_plot.fig.layout.shapes[0]
    assert shape.x0 == shape.x1 == x
    assert shape.y0 == min_val
    assert shape.y1 == max_val


def test_plot_milestone_marker_default_style(basic_line_plot):
    """Tests milestone marker with default style settings."""
    basic_line_plot._plot_milestone_marker(x=50, df_val_min=0, df_val_max=100)

    shape = basic_line_plot.fig.layout.shapes[0]
    assert shape.line.width == 1
    assert shape.line.color == "teal"


def test_save_html(basic_line_plot, tmp_path):
    """Tests saving plot as HTML file."""
    save_path = tmp_path / "test_plot.html"
    basic_line_plot.save(save_path)

    assert save_path.exists()
    assert save_path.suffix == ".html"
    assert save_path.stat().st_size > 0


@pytest.mark.skipif(not kaleido_available(), reason="kaleido package not installed")
def test_save_image(basic_line_plot, tmp_path):
    """Tests saving plot as image file."""
    save_path = tmp_path / "test_plot.png"
    basic_line_plot.save(save_path)

    assert save_path.exists()
    assert save_path.suffix == ".png"
    assert save_path.stat().st_size > 0


def test_save_no_figure(empty_plotter, tmp_path):
    """Tests error handling when saving without a figure."""
    save_path = tmp_path / "test_plot.html"

    with pytest.raises(LexosException) as exc_info:
        empty_plotter.save(save_path)
    assert "There is no plot to save" in str(exc_info.value)

@pytest.mark.skipif(not kaleido_available(), reason="kaleido package not installed")
@pytest.mark.parametrize("file_format", ["png", "jpg", "svg", "pdf"])
def test_save_different_formats(basic_line_plot, tmp_path, file_format):
    """Tests saving plot in different file formats.

    Args:
        file_format (str): File format to test
    """
    save_path = tmp_path / f"test_plot.{file_format}"
    basic_line_plot.save(save_path)

    assert save_path.exists()
    assert save_path.suffix == f".{file_format}"
    assert save_path.stat().st_size > 0


def test_save_with_kwargs(basic_line_plot, tmp_path):
    """Tests saving with additional kwargs for HTML output."""
    save_path = tmp_path / "test_plot.html"
    basic_line_plot.save(save_path, include_plotlyjs=True, full_html=True)

    assert save_path.exists()
    with open(save_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "plotly" in content
        assert "<html>" in content


def test_save_path_types(basic_line_plot, tmp_path):
    """Tests saving with different path types."""
    # Test with string path
    str_path = str(tmp_path / "test_plot_str.html")
    basic_line_plot.save(str_path)
    assert Path(str_path).exists()

    # Test with Path object
    path_obj = tmp_path / "test_plot_path.html"
    basic_line_plot.save(path_obj)
    assert path_obj.exists()


def test_show_basic(basic_line_plot):
    """Tests basic plot display with default config."""
    with patch.object(basic_line_plot.fig, "show") as mock_show:
        basic_line_plot.show()
        mock_show.assert_called_once_with(config={"displaylogo": False})


def test_show_custom_config(basic_line_plot):
    """Tests plot display with custom config."""
    custom_config = {"displaylogo": True, "scrollZoom": True, "displayModeBar": True}
    with patch.object(basic_line_plot.fig, "show") as mock_show:
        basic_line_plot.show(config=custom_config)
        mock_show.assert_called_once_with(config=custom_config)


def test_show_with_kwargs(basic_line_plot):
    """Tests plot display with additional kwargs."""
    with patch.object(basic_line_plot.fig, "show") as mock_show:
        basic_line_plot.show(config={"displaylogo": False})
        mock_show.assert_called_once_with(config={"displaylogo": False})


def test_show_no_logo_default(basic_line_plot):
    """Tests that displaylogo is False by default."""
    with patch.object(basic_line_plot.fig, "show") as mock_show:
        basic_line_plot.show()
        args = mock_show.call_args[1]
        assert args["config"]["displaylogo"] is False


@pytest.mark.parametrize(
    "config",
    [
        {"displaylogo": True},
        {"scrollZoom": True},
        {"displayModeBar": True},
        {"showLink": False},
        {},
    ],
)
def test_show_different_configs(basic_line_plot, config):
    """Tests plot display with various config options.

    Args:
        config (dict): Configuration dictionary to test
    """
    with patch.object(basic_line_plot.fig, "show") as mock_show:
        basic_line_plot.show(config=config)
        mock_show.assert_called_once_with(config=config)
