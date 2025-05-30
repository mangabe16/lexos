"""test_simple_plotter.py.

Last Updated: February 10, 2025
"""

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend to avoid TclError
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# from matplotlib.axes import Axes
from lexos.exceptions import LexosException
from lexos.rolling_windows.plotters.simple_plotter import SimplePlotter

# Fixtures


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({"term1": [1, 2, 3, 4, 5], "term2": [2, 4, 6, 8, 10]})


@pytest.fixture
def basic_plotter():
    """Create a SimplePlotter instance with default settings."""
    return SimplePlotter()


@pytest.fixture
def milestone_plotter():
    """Create a SimplePlotter instance with milestone settings."""
    return SimplePlotter(
        show_milestones=True, milestone_labels={"Chapter 1": 100, "Chapter 2": 200}
    )


@pytest.fixture
def plotter_with_plot(sample_df):
    """Create SimplePlotter instance with an existing plot.

    Args:
        sample_df: Sample DataFrame fixture

    Returns:
        SimplePlotter: Configured plotter instance with plot
    """
    plotter = SimplePlotter()
    plotter(df=sample_df)
    return plotter


@pytest.fixture
def sample_df2():
    """Create sample DataFrame for testing.

    Returns:
        pd.DataFrame: DataFrame with test data
    """
    return pd.DataFrame({"term1": [1, 4, 2, 5, 3], "term2": [2, 5, 3, 6, 4]})


@pytest.fixture
def interpolation_plotter():
    """Create SimplePlotter instance with interpolation settings.

    Returns:
        SimplePlotter: Configured plotter instance
    """
    return SimplePlotter(
        use_interpolation=True, interpolation_num=100, interpolation_kind="linear"
    )


@pytest.fixture
def sample_df3():
    """Create sample DataFrame for testing.

    Returns:
        pd.DataFrame: Sample DataFrame with test data
    """
    return pd.DataFrame({"term1": [1, 2, 3], "term2": [2, 4, 6]})


@pytest.fixture
def plotter_with_plot2(sample_df3):
    """Create SimplePlotter instance with an existing plot.

    Returns:
        SimplePlotter: Configured plotter instance with plot
    """
    plotter = SimplePlotter()
    plotter(df=sample_df3)
    return plotter


@pytest.fixture
def milestone_plotter2():
    """Create SimplePlotter instance with milestone settings.

    Returns:
        SimplePlotter: Configured plotter instance
    """
    return SimplePlotter(
        show_milestones=True,
        show_milestone_labels=True,
        milestone_labels={"Start": 0, "Middle": 2, "End": 4},
        milestone_colors="red",
        milestone_style="--",
        milestone_width=2,
    )


@pytest.fixture
def sample_milestone_labels():
    """Create sample milestone labels for testing.

    Returns:
        dict: Sample milestone labels and positions
    """
    return {"Short": 0, "Medium Label": 100, "Very Long Label Text": 200}


@pytest.fixture
def milestone_plotter3():
    """Creates a SimplePlotter instance with milestone settings.

    Returns:
        SimplePlotter: A plotter instance configured with milestone labels
    """
    return SimplePlotter(
        show_milestone_labels=True,
        milestone_labels={"Test1": 0, "Test2": 100},
        title_position="top",
    )


# Tests


def test_default_initialization(basic_plotter):
    """Test SimplePlotter initialization with default values."""
    assert basic_plotter.id == "rw_simple_plotter"
    assert basic_plotter.width == 6.4
    assert basic_plotter.height == 4.8
    assert basic_plotter.hide_spines == ["top", "right"]
    assert basic_plotter.title == "Rolling Windows Plot"


def test_custom_initialization():
    """Test SimplePlotter initialization with custom values."""
    custom_plotter = SimplePlotter(
        width=10.0, height=8.0, title="Custom Plot", show_grid=True
    )
    assert custom_plotter.width == 10.0
    assert custom_plotter.height == 8.0
    assert custom_plotter.title == "Custom Plot"
    assert custom_plotter.show_grid is True


def test_figsize_override():
    """Test that figsize parameter overrides width and height."""
    plotter = SimplePlotter(width=10.0, height=8.0, figsize=(12.0, 6.0))
    width, height = plotter._get_width_height()
    assert width == 12.0
    assert height == 6.0


def test_milestone_validation(basic_plotter):
    """Test milestone validation with invalid settings."""
    with pytest.raises(Exception):
        basic_plotter.show_milestones = True
        basic_plotter._validate_edge_cases()


def test_valid_milestone_settings(milestone_plotter):
    """Test valid milestone settings."""
    milestone_plotter._validate_edge_cases()
    assert milestone_plotter.milestone_labels == {"Chapter 1": 100, "Chapter 2": 200}


@pytest.mark.parametrize("spine_setting", [["top", "right"], ["all"], ["bottom"], []])
def test_hide_spines_settings(spine_setting):
    """Test various spine hiding configurations."""
    plotter = SimplePlotter(hide_spines=spine_setting)
    assert plotter.hide_spines == spine_setting


def test_interpolation_settings():
    """Test interpolation settings."""
    plotter = SimplePlotter(
        use_interpolation=True, interpolation_num=1000, interpolation_kind="cubic"
    )
    assert plotter.use_interpolation is True
    assert plotter.interpolation_num == 1000
    assert plotter.interpolation_kind == "cubic"


def test_milestone_color_settings():
    """Test milestone color settings."""
    single_color = SimplePlotter(milestone_colors="red")
    assert single_color.milestone_colors == "red"

    multi_color = SimplePlotter(milestone_colors=["red", "blue"])
    assert multi_color.milestone_colors == ["red", "blue"]


def test_call_basic_plot(basic_plotter, sample_df):
    """Test basic plot creation with default settings."""
    basic_plotter(df=sample_df)
    assert isinstance(basic_plotter.fig, plt.Figure)
    plt.close()


def test_call_with_custom_dimensions(basic_plotter, sample_df):
    """Test plot creation with custom dimensions."""
    basic_plotter(df=sample_df, width=10, height=8)
    fig = basic_plotter.fig
    assert fig.get_size_inches()[0] == 10
    assert fig.get_size_inches()[1] == 8
    plt.close()


def test_call_with_figsize(basic_plotter, sample_df):
    """Test plot creation with figsize override."""
    basic_plotter(df=sample_df, figsize=(12, 6))
    fig = basic_plotter.fig
    assert fig.get_size_inches()[0] == 12
    assert fig.get_size_inches()[1] == 6
    plt.close()


def test_call_with_grid(basic_plotter, sample_df):
    """Test plot creation with grid enabled."""
    basic_plotter(df=sample_df, show_grid=True)
    assert basic_plotter.fig.axes[0].get_xgridlines()
    plt.close()


def test_call_with_interpolation(basic_plotter, sample_df):
    """Test plot creation with interpolation."""
    basic_plotter(
        df=sample_df,
        use_interpolation=True,
        interpolation_num=1000,
        interpolation_kind="cubic",
    )
    assert basic_plotter.fig is not None
    plt.close()


def test_call_with_milestones(basic_plotter, sample_df):
    """Test plot creation with milestones."""
    milestone_labels = {"Chapter 1": 2, "Chapter 2": 4}
    basic_plotter(df=sample_df, show_milestones=True, milestone_labels=milestone_labels)
    assert basic_plotter.fig is not None
    plt.close()


def test_call_with_milestone_labels(basic_plotter, sample_df):
    """Test plot creation with milestone labels."""
    milestone_labels = {"Test 1": 2, "Test 2": 4}
    basic_plotter(
        df=sample_df,
        show_milestone_labels=True,
        milestone_labels=milestone_labels,
        milestone_labels_rotation=90,
    )
    assert basic_plotter.fig is not None
    plt.close()


def test_call_invalid_df():
    """Test plot creation with invalid DataFrame."""
    plotter = SimplePlotter()
    with pytest.raises(Exception):
        plotter(df=None)


def test_call_show_plot_false(basic_plotter, sample_df):
    """Test plot creation with show_plot=False."""
    basic_plotter(df=sample_df, show_plot=False)
    assert basic_plotter.fig is not None
    plt.close()


def test_call_custom_styles(basic_plotter, sample_df):
    """Test plot creation with custom styling."""
    basic_plotter(
        df=sample_df,
        title="Custom Title",
        xlabel="Custom X",
        ylabel="Custom Y",
        hide_spines=["top", "right", "bottom"],
        title_position="bottom",
    )
    fig = basic_plotter.fig
    ax = fig.axes[0]
    assert ax.get_title() == "Custom Title"
    assert ax.get_xlabel() == "Custom X"
    assert ax.get_ylabel() == "Custom Y"
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert not ax.spines["bottom"].get_visible()
    plt.close()


def test_adjust_titlepad_default(basic_plotter):
    """Tests titlepad adjustment with default settings."""
    result = basic_plotter._adjust_titlepad(6.0, 6.4, 4.8)
    assert result == 6.0
    plt.close()


def test_adjust_titlepad_custom_value(basic_plotter):
    """Tests titlepad adjustment with custom padding."""
    custom_pad = 10.0
    result = basic_plotter._adjust_titlepad(custom_pad, 6.4, 4.8)
    assert result == custom_pad
    plt.close()


def test_adjust_titlepad_with_milestones(milestone_plotter3):
    """Tests titlepad adjustment with milestone labels enabled."""
    result = milestone_plotter3._adjust_titlepad(6.0, 6.4, 4.8)
    assert result > 6.0  # Should be adjusted for milestone labels
    plt.close()


def test_adjust_titlepad_bottom_title(milestone_plotter3):
    """Tests titlepad adjustment with bottom title position."""
    milestone_plotter3.title_position = "bottom"
    result = milestone_plotter3._adjust_titlepad(6.0, 6.4, 4.8)
    assert result == 6.0  # Should not adjust for bottom titles
    plt.close()


def test_adjust_titlepad_custom_milestone_pad(milestone_plotter3):
    """Tests titlepad adjustment with custom padding and milestones."""
    milestone_plotter3.titlepad = 15.0
    result = milestone_plotter3._adjust_titlepad(15.0, 6.4, 4.8)
    assert result == 15.0  # Should not adjust for custom padding


def test_get_label_height_basic(basic_plotter, sample_milestone_labels):
    """Test basic label height calculation."""
    height = basic_plotter._get_label_height(
        sample_milestone_labels, milestone_labels_rotation=45
    )
    assert isinstance(height, float)
    assert height > 0
    plt.close()


@pytest.mark.parametrize("rotation", [0, 45, 90])
def test_get_label_height_rotations(basic_plotter, sample_milestone_labels, rotation):
    """Test label height calculation with different rotations.

    Args:
        rotation (int): Rotation angle to test
    """
    height = basic_plotter._get_label_height(
        sample_milestone_labels, milestone_labels_rotation=rotation
    )
    assert isinstance(height, float)
    assert height > 0
    plt.close()


def test_get_label_height_empty_labels(basic_plotter):
    """Test label height calculation with empty labels dictionary."""
    with pytest.raises(ValueError):
        basic_plotter._get_label_height({}, milestone_labels_rotation=45)
    plt.close()


def test_get_label_height_single_label(basic_plotter):
    """Test label height calculation with single label."""
    height = basic_plotter._get_label_height(
        {"Single": 0}, milestone_labels_rotation=45
    )
    assert isinstance(height, float)
    assert height > 0
    plt.close()


def test_get_label_height_unicode(basic_plotter):
    """Test label height calculation with Unicode characters."""
    unicode_labels = {"αβγ": 0, "漢字": 100, "🌟✨": 200}
    height = basic_plotter._get_label_height(
        unicode_labels, milestone_labels_rotation=45
    )
    assert isinstance(height, float)
    assert height > 0
    plt.close()


def test_get_label_height_comparison(basic_plotter):
    """Test that longer labels produce greater heights."""
    short_label = {"A": 0}
    long_label = {"Very Long Label Text": 0}

    short_height = basic_plotter._get_label_height(
        short_label, milestone_labels_rotation=0
    )
    long_height = basic_plotter._get_label_height(
        long_label, milestone_labels_rotation=0
    )

    assert long_height >= short_height
    plt.close()


def test_plot_interpolated_basic(interpolation_plotter, sample_df2):
    """Test basic interpolation plotting functionality."""
    fig, ax = plt.subplots()
    interpolation_plotter._plot_interpolated(sample_df2)

    # Check if lines were plotted
    assert len(ax.get_lines()) == 2  # Lines are on different axes
    plt.close()


def test_plot_interpolated_custom_style(interpolation_plotter, sample_df2):
    """Test interpolation plotting with custom style parameters."""
    fig, ax = plt.subplots()
    interpolation_plotter._plot_interpolated(sample_df2, linestyle="--", color="red")
    plt.close()


def test_plot_interpolated_empty_df(interpolation_plotter):
    """Test interpolation plotting with empty DataFrame."""
    empty_df = pd.DataFrame()
    fig, ax = plt.subplots()
    with pytest.raises(IndexError):
        interpolation_plotter._plot_interpolated(empty_df)
    plt.close()


@pytest.mark.parametrize("interp_kind", ["linear", "cubic", "pchip"])
def test_plot_interpolated_methods(sample_df2, interp_kind):
    """Test different interpolation methods.

    Args:
        interp_kind (str): Interpolation method to test
    """
    plotter = SimplePlotter(
        use_interpolation=True, interpolation_num=100, interpolation_kind=interp_kind
    )
    fig, ax = plt.subplots()
    plotter._plot_interpolated(sample_df2)
    plt.close()


def test_plot_interpolated_num_points(sample_df2):
    """Test number of interpolation points."""
    num_points = 1000
    plotter = SimplePlotter(
        use_interpolation=True,
        interpolation_num=num_points,
        interpolation_kind="linear",
    )
    fig, ax = plt.subplots()
    plotter._plot_interpolated(sample_df2)

    # Get the x values from the plotted line
    lines = plt.gca().get_lines()
    assert len(lines) > 0
    x_data = lines[0].get_xdata()
    assert len(x_data) == num_points

    plt.close()


def test_plot_interpolated_data_preservation(interpolation_plotter, sample_df2):
    """Test that original data points are preserved in interpolation."""
    fig, ax = plt.subplots()
    interpolation_plotter._plot_interpolated(sample_df2)

    # Get the interpolated values
    lines = plt.gca().get_lines()
    assert len(lines) > 0
    y_interpolated = lines[0].get_ydata()

    # Check if original points are approximately preserved
    original_points = sample_df2["term1"].values
    x_original = np.arange(len(original_points))
    x_interpolated = lines[0].get_xdata()

    # Find indices in interpolated data corresponding to original points
    for i, x in enumerate(x_original):
        idx = np.abs(x_interpolated - x).argmin()
        # This works for the test data, but it is not generalisable
        assert round(y_interpolated[idx]) == original_points[i]
        # assert np.isclose(y_interpolated[idx], original_points[i], atol=1e-15)

    plt.close()


def test_show_milestones_basic(milestone_plotter2, sample_df3):
    """Test basic milestone plotting functionality."""
    fig, ax = plt.subplots()
    milestone_plotter2._show_milestones(sample_df3, ax)

    # Check if vertical lines were added
    lines = [
        child
        for child in ax.get_children()
        if isinstance(child, matplotlib.collections.LineCollection)
    ]
    assert len(lines) == 3  # Should have 3 milestone lines

    # Check if annotations were added
    annotations = ax.get_children()
    text_annotations = [a for a in annotations if isinstance(a, plt.Text)]
    assert len(text_annotations) == 6  # Should have 3 milestone labels and 3 ticks

    plt.close()


def test_show_milestones_only_markers(sample_df3):
    """Test milestone plotting with only markers (no labels)."""
    plotter = SimplePlotter(
        show_milestones=True,
        show_milestone_labels=False,
        milestone_labels={"A": 1, "B": 3},
    )
    fig, ax = plt.subplots()
    plotter._show_milestones(sample_df3, ax)

    # Check vertical lines
    lines = [
        child
        for child in ax.get_children()
        if isinstance(child, matplotlib.collections.LineCollection)
    ]
    assert len(lines) == 2

    # Check no annotations
    annotations = ax.get_children()
    text_annotations = [
        a for a in annotations if isinstance(a, plt.Text) and a.get_text() != ""
    ]
    assert len(text_annotations) == 0

    plt.close()


def test_show_milestones_only_labels(sample_df3):
    """Test milestone plotting with only labels (no markers)."""
    plotter = SimplePlotter(
        show_milestones=False,
        show_milestone_labels=True,
        milestone_labels={"A": 1, "B": 3},
    )
    fig, ax = plt.subplots()
    plotter._show_milestones(sample_df3, ax)

    # Check no vertical lines
    lines = [
        child
        for child in ax.get_children()
        if isinstance(child, matplotlib.collections.LineCollection)
    ]
    assert len(lines) == 0

    # Check annotations
    annotations = ax.get_children()
    text_annotations = [
        a for a in annotations if isinstance(a, plt.Text) and len(a.get_text()) > 0
    ]
    assert len(text_annotations) == 2

    plt.close()


def test_show_milestones_custom_style(sample_df3):
    """Test milestone plotting with custom styling."""
    plotter = SimplePlotter(
        show_milestones=True,
        show_milestone_labels=True,
        milestone_labels={"Test": 2},
        milestone_colors="0.8",
        milestone_style=":",
        milestone_width=3,
        milestone_labels_rotation=90,
        milestone_labels_ha="right",
        milestone_labels_va="top",
    )
    fig, ax = plt.subplots()
    plotter._show_milestones(sample_df3, ax)

    # Check style of vertical lines
    lines = [
        child
        for child in ax.get_children()
        if isinstance(child, matplotlib.collections.LineCollection)
    ]
    assert len(lines) == 1
    # Confirm that the colour is not the default -- there doesn't seem to be an easy way to detect
    # a specific colour.
    set_colors = list(lines[0].get_colors()[0])
    assert set_colors != [0.0, 0.50196078, 0.50196078, 1.0]
    assert lines[0].get_linewidth() == 3

    # Check annotation styling
    annotations = [a for a in ax.get_children() if isinstance(a, plt.Text)]
    assert annotations[0].get_rotation() == 90
    assert annotations[0].get_ha() == "right"
    assert annotations[0].get_va() == "top"

    plt.close()


def test_show_milestones_ymax_calculation(sample_df3):
    """Test that milestone heights use correct max y-value."""
    plotter = SimplePlotter(show_milestones=True, milestone_labels={"Test": 2})
    fig, ax = plt.subplots()
    plotter._show_milestones(sample_df3, ax)

    # Get the vertical line
    # lines = [line for line in ax.get_lines() if line.get_linestyle() == '--']
    lines = [
        child
        for child in ax.get_children()
        if isinstance(child, matplotlib.collections.LineCollection)
    ]
    assert len(lines) == 1

    # Check line extends from 0 to max value in DataFrame
    ydata = lines[0].get_segments()
    min_y = ydata[0][0][1]  # 0.0
    max_y = ydata[0][1][1]  # 10.0
    assert min_y == 0
    assert max_y == sample_df3.to_numpy().max()

    plt.close()


def test_save_basic(plotter_with_plot, tmp_path):
    """Test basic plot saving functionality.

    Args:
        plotter_with_plot: Plotter fixture with existing plot
        tmp_path: pytest temporary path fixture
    """
    save_path = tmp_path / "test_plot.png"
    plotter_with_plot.save(save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_save_no_plot():
    """Test saving without creating plot first."""
    plotter = SimplePlotter()
    with pytest.raises(LexosException) as exc_info:
        plotter.save("test.png")
    assert "There is no plot to save" in str(exc_info.value)


def test_save_different_formats(plotter_with_plot, tmp_path):
    """Test saving plot in different formats.

    Args:
        plotter_with_plot: Plotter fixture with existing plot
        tmp_path: pytest temporary path fixture
    """
    formats = [".png", ".pdf", ".jpg", ".svg"]
    for fmt in formats:
        save_path = tmp_path / f"test_plot{fmt}"
        plotter_with_plot.save(save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


def test_save_with_kwargs(plotter_with_plot, tmp_path):
    """Test saving plot with additional kwargs.

    Args:
        plotter_with_plot: Plotter fixture with existing plot
        tmp_path: pytest temporary path fixture
    """
    save_path = tmp_path / "test_plot_dpi.png"
    plotter_with_plot.save(save_path, dpi=300, bbox_inches="tight")
    assert save_path.exists()

    # Higher DPI should result in larger file size
    save_path_low_dpi = tmp_path / "test_plot_low_dpi.png"
    plotter_with_plot.save(save_path_low_dpi, dpi=72)
    assert save_path.stat().st_size > save_path_low_dpi.stat().st_size


def test_save_path_types(plotter_with_plot, tmp_path):
    """Test saving with different path types.

    Args:
        plotter_with_plot: Plotter fixture with existing plot
        tmp_path: pytest temporary path fixture
    """
    # Test with string path
    str_path = str(tmp_path / "test_plot_str.png")
    plotter_with_plot.save(str_path)
    assert Path(str_path).exists()

    # Test with Path object
    path_obj = tmp_path / "test_plot_path.png"
    plotter_with_plot.save(path_obj)
    assert path_obj.exists()


def test_show_no_plot():
    """Test showing plot without creating one first."""
    plotter = SimplePlotter()
    with pytest.raises(LexosException) as exc_info:
        plotter.show()
    assert "There is no plot to show" in str(exc_info.value)


def test_show_with_plot(plotter_with_plot2):
    """Test showing plot with existing figure."""
    with patch.object(plotter_with_plot2.fig, "show") as mock_show:
        plotter_with_plot2.show()
        mock_show.assert_called_once()


def test_show_with_kwargs(plotter_with_plot2):
    """Test showing plot with additional kwargs."""
    test_kwargs = {"block": False}
    with patch.object(plotter_with_plot2.fig, "show") as mock_show:
        plotter_with_plot2.show(**test_kwargs)
        mock_show.assert_called_once_with(**test_kwargs)


def test_show_jupyter_fallback(plotter_with_plot2):
    """Test fallback behavior for Jupyter notebooks."""
    with patch.object(plotter_with_plot2.fig, "show", side_effect=UserWarning):
        result = plotter_with_plot2.show()
        assert result == plotter_with_plot2.fig

# interpolation tests

def test_interpolate_numpy_fallback():
    """Test interpolate function with unrecognized interpolation_kind (line 61 coverage)."""
    from lexos.rolling_windows.plotters.simple_plotter import interpolate
    
    # Create test data
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 2, 4, 6, 8])
    xx = np.array([0.5, 1.5, 2.5, 3.5])
    
    # Use an interpolation_kind that's not "pchip" or in legacy_interp1d list
    # This should trigger the else clause on line 61
    result = interpolate(x, y, xx, interpolation_kind="unknown_method")
    
    # Verify it returns results (using np.interp as fallback)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(xx)
    
    # Expected values from np.interp
    expected = np.interp(xx, x, y)
    np.testing.assert_array_equal(result, expected)

def test_interpolate_none_interpolation_kind():
    """Test interpolate function with None interpolation_kind (line 61 coverage)."""
    from lexos.rolling_windows.plotters.simple_plotter import interpolate
    
    # Create test data
    x = np.array([0, 1, 2, 3])
    y = np.array([1, 3, 5, 7])
    xx = np.array([0.5, 1.5, 2.5])
    
    # Use None as interpolation_kind - should trigger else clause on line 61
    result = interpolate(x, y, xx, interpolation_kind=None)
    
    # Verify it returns results using np.interp
    assert isinstance(result, np.ndarray)
    assert len(result) == len(xx)
    
    # Expected values from np.interp
    expected = np.interp(xx, x, y)
    np.testing.assert_array_equal(result, expected)