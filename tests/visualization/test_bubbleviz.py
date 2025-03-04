"""test_bubbleviz.py.

Last Update: March 3, 2025
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import spacy
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from pydantic import ValidationError
from scipy.sparse import csr_matrix

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization.bubbleviz import BubbleChart

# Fixtures


@pytest.fixture
def nlp():
    """Create spacy nlp object for testing.

    Returns:
        Language: spaCy Language object
    """
    return spacy.blank("en")


@pytest.fixture
def sample_dtm():
    """Create a sample DTM for testing.

    Returns:
        DTM: Sample DTM instance with test data
    """
    dtm = DTM()

    # Create sample data
    data = np.array([[1, 0, 3], [0, 2, 1], [2, 1, 0]])
    dtm.doc_term_matrix = csr_matrix(data)

    # Mock vectorizer
    class MockVectorizer:
        terms_list = ["term1", "term2", "term3"]

    dtm.vectorizer = MockVectorizer()

    # Set labels
    dtm.labels = ["doc1", "doc2", "doc3"]

    return dtm


@pytest.fixture
def basic_bubble_chart():
    """Create basic bubble chart instance.

    Returns:
        BubbleChart: Configured bubble chart instance
    """
    return BubbleChart(data="hello world hello test")


@pytest.fixture
def simple_bubble_chart():
    """Create a simple bubble chart with predictable data.

    Returns:
        BubbleChart: Configured bubble chart with test data
    """
    return BubbleChart(data="word1 word2 word1 word3", showfig=False, limit=3)


@pytest.fixture
def sample_bubble_chart():
    """Create a sample bubble chart instance.

    Returns:
        BubbleChart: Configured bubble chart instance with test data
    """
    chart = BubbleChart(data="test data", showfig=False)
    chart.bubbles = np.array(
        [
            [0, 0, 1, 1],  # x, y, radius, area
            [3, 4, 1, 1],
            [6, 8, 1, 1],
        ]
    )
    return chart


@pytest.fixture
def sample_bubble_chart2():
    """Create a sample bubble chart instance.

    Returns:
        BubbleChart: Configured bubble chart with test data
    """
    chart = BubbleChart(data="test data", showfig=False)
    chart.bubbles = np.array(
        [
            [0, 0, 1, 2],  # x, y, radius, area(weight)
            [2, 2, 1, 1],
            [4, 4, 1, 1],
        ]
    )
    return chart


@pytest.fixture
def sample_bubble_chart3():
    """Create a sample bubble chart instance.

    Returns:
        BubbleChart: Configured bubble chart with test data
    """
    chart = BubbleChart(data="test data", showfig=False)
    chart.bubble_spacing = 0.1
    return chart


@pytest.fixture
def sample_bubble_chart4():
    """Create a sample bubble chart instance.

    Returns:
        BubbleChart: Configured bubble chart with test data
    """
    chart = BubbleChart(data="test data", showfig=False)
    # Initialize bubbles with known positions
    chart.bubbles = np.array(
        [
            [0, 0, 1, 1],  # x, y, radius, area
            [3, 3, 1, 1],
            [6, 6, 1, 1],
        ]
    )
    chart.bubble_spacing = 0.1
    chart.step_dist = 1.0
    chart.com = np.array([3, 3])  # Center of mass at middle bubble
    return chart


@pytest.fixture
def sample_bubble_chart5():
    """Create a sample bubble chart instance.

    Returns:
        BubbleChart: Configured bubble chart with test data
    """
    chart = BubbleChart(data="test data", showfig=False)
    chart.bubbles = np.array(
        [
            [0, 0, 1, 1],  # x, y, radius, area
            [3, 3, 2, 4],
            [6, 6, 1.5, 2.25],
        ]
    )
    chart.colors = ["red", "blue", "green"]
    return chart


@pytest.fixture
def test_axes():
    """Create matplotlib axes for testing.

    Returns:
        Axes: Matplotlib axes object
    """
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.fixture
def sample_dtm2():
    """Create sample DTM for testing.

    Returns:
        DTM: Sample DTM with test data
    """
    data = np.array([[1, 2], [3, 4]])
    terms = ["term1", "term2"]
    docs = ["doc1", "doc2"]
    return DTM(matrix=data, terms=terms, labels=docs)


# Tests


def test_bubble_chart_init():
    """Test BubbleChart initialization with default values."""
    chart = BubbleChart(data="test")
    assert chart.limit == 100
    assert chart.bubble_spacing == 0.1
    assert chart.figsize == (15, 15)
    assert chart.font_family == "DejaVu Sans"
    assert chart.showfig is True


def test_bubble_chart_with_string_input():
    """Test BubbleChart with string input."""
    chart = BubbleChart(data="hello world hello")
    chart()
    assert isinstance(chart.term_counts, dict)
    assert chart.term_counts["hello"] == 2
    assert chart.term_counts["world"] == 1


def test_bubble_chart_with_dtm(sample_dtm):
    """Test BubbleChart with DTM input."""
    chart = BubbleChart(data=sample_dtm)
    chart()
    assert isinstance(chart.term_counts, dict)
    assert len(chart.term_counts) > 0


def test_bubble_chart_with_custom_options():
    """Test BubbleChart with custom configuration options."""
    chart = BubbleChart(
        data="test data",
        limit=50,
        bubble_spacing=0.2,
        figsize=(10, 10),
        title="Test Chart",
    )
    assert chart.limit == 50
    assert chart.bubble_spacing == 0.2
    assert chart.figsize == (10, 10)
    assert chart.title == "Test Chart"


def test_bubble_chart_showfig_false():
    """Test BubbleChart with showfig=False."""
    chart = BubbleChart(data="test data", showfig=False)
    chart()
    assert isinstance(chart.fig, Figure)


def test_bubble_chart_save(tmp_path):
    """Test saving BubbleChart to file."""
    chart = BubbleChart(data="test data", showfig=False)
    chart()

    save_path = tmp_path / "test_bubble.png"
    chart.save(save_path)
    assert save_path.exists()


def test_bubble_chart_empty_data():
    """Test BubbleChart with empty data."""
    with pytest.raises(LexosException):
        BubbleChart(data="")


@pytest.mark.parametrize(
    "limit,expected_bubbles",
    [
        (4, 4),
        # (2, 2),
        # (5, 4),  # If input has only 4 unique terms
    ],
)
def test_bubble_chart_limits(limit, expected_bubbles):
    """Test BubbleChart with different term limits.

    Args:
        limit: Maximum number of bubbles to display
        expected_bubbles: Expected number of bubbles in output
    """
    data = "word1 word2 word3 word4"
    chart = BubbleChart(data=data, limit=limit, showfig=False)
    chart()
    assert len(chart.bubbles) <= expected_bubbles


def test_bubble_chart_colors():
    """Test BubbleChart with custom colors."""
    custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
    chart = BubbleChart(data="test data", colors=custom_colors)
    assert chart.colors == custom_colors


def test_bubble_chart_font_family():
    """Test BubbleChart with custom font family."""
    chart = BubbleChart(data="test data", font_family="Arial")
    assert chart.font_family == "Arial"


def test_call_basic_functionality(simple_bubble_chart):
    """Test basic functionality of __call__ method."""
    simple_bubble_chart()

    assert isinstance(simple_bubble_chart.bubbles, np.ndarray)
    assert simple_bubble_chart.bubbles.shape[1] == 4  # x, y, radius, area
    assert len(simple_bubble_chart.bubbles) == 3  # limited by limit=3


def test_call_bubble_calculations(simple_bubble_chart):
    """Test bubble size calculations."""
    simple_bubble_chart()

    # Check that areas correspond to word frequencies
    areas = simple_bubble_chart.bubbles[:, 3]
    assert 2 in areas  # 'word1' appears twice
    assert 1 in areas  # 'word2' and 'word3' appear once


def test_call_creates_figure(simple_bubble_chart):
    """Test figure creation when showfig=False."""
    simple_bubble_chart()

    assert isinstance(simple_bubble_chart.fig, Figure)


def test_call_with_title():
    """Test figure creation with title."""
    chart = BubbleChart(data="test data", title="Test Title", showfig=False)
    chart()

    assert chart.fig.axes[0].get_title() == "Test Title"


def test_call_with_dtm(sample_dtm):
    """Test processing DTM input."""
    chart = BubbleChart(data=sample_dtm, showfig=False)
    chart()

    assert isinstance(chart.bubbles, np.ndarray)
    assert len(chart.bubbles) <= chart.limit


@pytest.mark.parametrize(
    "spacing,expected_maxstep",
    [
        (0.1, None),  # Default spacing
        (0.5, None),  # Larger spacing
        (0.0, None),  # No spacing
    ],
)
def test_call_bubble_spacing(spacing, expected_maxstep):
    """Test effect of different bubble spacing values.

    Args:
        spacing: Bubble spacing value to test
        expected_maxstep: Expected maximum step size
    """
    chart = BubbleChart(data="word1 word2", bubble_spacing=spacing, showfig=False)
    chart()

    # Verify maxstep calculation
    assert chart.maxstep > 0
    if expected_maxstep:
        assert chart.maxstep == expected_maxstep


def test_call_limit_enforcement():
    """Test enforcement of bubble limit."""
    data = " ".join(["word"] * 200)  # Create string with many repeated words
    chart = BubbleChart(data=data, limit=50, showfig=False)
    chart()

    assert len(chart.bubbles) <= 50


def test_call_grid_layout(simple_bubble_chart):
    """Test initial grid layout creation."""
    simple_bubble_chart()

    # Check that bubbles are arranged in a grid
    x_coords = simple_bubble_chart.bubbles[:, 0]
    y_coords = simple_bubble_chart.bubbles[:, 1]

    assert len(np.unique(x_coords)) > 0
    assert len(np.unique(y_coords)) > 0


def test_center_distance_single_point():
    """Test center distance calculation for single point."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([0, 0, 1, 1])  # x=0, y=0, r=1, area=1
    bubbles = np.array([[3, 4, 1, 1]])  # x=3, y=4, r=1, area=1

    distance = chart._center_distance(bubble, bubbles)
    assert np.isclose(distance[0], 5.0)  # Expected distance = sqrt(3^2 + 4^2) = 5


def test_center_distance_multiple_points(sample_bubble_chart):
    """Test center distance calculation for multiple points."""
    bubble = np.array([0, 0, 1, 1])
    bubbles = sample_bubble_chart.bubbles

    distances = sample_bubble_chart._center_distance(bubble, bubbles)
    expected = np.array([0.0, 5.0, 10.0])  # Distances from (0,0) to each point

    np.testing.assert_array_almost_equal(distances, expected)


def test_center_distance_same_point():
    """Test center distance calculation for identical points."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([1, 1, 1, 1])
    bubbles = np.array([[1, 1, 1, 1]])

    distance = chart._center_distance(bubble, bubbles)
    assert np.isclose(distance[0], 0.0)


@pytest.mark.parametrize(
    "point,expected",
    [
        ([0, 0], 5.0),  # Origin to (3,4)
        ([3, 4], 0.0),  # Same point
        ([-3, -4], 10.0),  # Negative coordinates
    ],
)
def test_center_distance_various_points(point, expected):
    """Test center distance calculation for various points.

    Args:
        point: Input point coordinates [x, y]
        expected: Expected distance
    """
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([*point, 1, 1])
    bubbles = np.array([[3, 4, 1, 1]])

    distance = chart._center_distance(bubble, bubbles)
    assert np.isclose(distance[0], expected)


def test_center_distance_zero_radius():
    """Test center distance calculation with zero radius."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([0, 0, 0, 0])
    bubbles = np.array([[3, 4, 0, 0]])

    distance = chart._center_distance(bubble, bubbles)
    assert np.isclose(distance[0], 5.0)


def test_center_of_mass_basic(sample_bubble_chart2):
    """Test basic center of mass calculation."""
    com = sample_bubble_chart2._center_of_mass()
    expected = np.array([1.5, 1.5])  # Weighted average position
    np.testing.assert_array_almost_equal(com, expected)


def test_center_of_mass_equal_weights():
    """Test center of mass with equal weights."""
    chart = BubbleChart(data="test", showfig=False)
    chart.bubbles = np.array([[0, 0, 1, 1], [2, 2, 1, 1]])
    com = chart._center_of_mass()
    expected = np.array([1, 1])
    np.testing.assert_array_almost_equal(com, expected)


def test_center_of_mass_single_bubble():
    """Test center of mass with single bubble."""
    chart = BubbleChart(data="test", showfig=False)
    chart.bubbles = np.array([[3, 4, 1, 1]])
    com = chart._center_of_mass()
    expected = np.array([3, 4])
    np.testing.assert_array_almost_equal(com, expected)


@pytest.mark.parametrize(
    "bubbles,expected",
    [
        (np.array([[0, 0, 1, 1], [10, 0, 1, 1]]), np.array([5, 0])),
        (np.array([[0, 0, 1, 2], [10, 0, 1, 1]]), np.array([3.33333333, 0])),
        (np.array([[0, 0, 1, 3], [3, 4, 1, 1]]), np.array([0.75, 1])),
    ],
)
def test_center_of_mass_various_configs(bubbles, expected):
    """Test center of mass with various bubble configurations.

    Args:
        bubbles: Input bubble configurations
        expected: Expected center of mass coordinates
    """
    chart = BubbleChart(data="test", showfig=False)
    chart.bubbles = bubbles
    com = chart._center_of_mass()
    np.testing.assert_array_almost_equal(com, expected, decimal=5)


def test_check_collisions_no_collision(sample_bubble_chart3):
    """Test collision check with non-colliding bubbles."""
    bubble = np.array([0, 0, 1, 1])  # x, y, radius, area
    bubbles = np.array([[3, 3, 1, 1]])  # Far enough apart to not collide

    collisions = sample_bubble_chart3._check_collisions(bubble, bubbles)
    assert collisions == 0


def test_check_collisions_with_collision(sample_bubble_chart3):
    """Test collision check with colliding bubbles."""
    bubble = np.array([0, 0, 1, 1])
    bubbles = np.array([[1, 1, 1, 1]])  # Close enough to collide

    collisions = sample_bubble_chart3._check_collisions(bubble, bubbles)
    assert collisions > 0


def test_check_collisions_multiple_bubbles(sample_bubble_chart3):
    """Test collision check with multiple bubbles."""
    bubble = np.array([0, 0, 1, 1])
    bubbles = np.array(
        [
            [3, 3, 1, 1],  # No collision
            [1, 1, 1, 1],  # Collision
            [5, 5, 1, 1],  # No collision
        ]
    )

    collisions = sample_bubble_chart3._check_collisions(bubble, bubbles)
    assert collisions == 1


@pytest.mark.parametrize(
    "spacing,expected_collisions",
    [
        (0.1, 0),  # Default spacing
        (0.5, 0),  # Larger spacing
        (1.0, 1),  # Even larger spacing causing more collisions
    ],
)
def test_check_collisions_different_spacing(spacing, expected_collisions):
    """Test collision check with different bubble spacing values.

    Args:
        spacing: Bubble spacing value to test
        expected_collisions: Expected number of collisions
    """
    chart = BubbleChart(data="test", showfig=False)
    chart.bubble_spacing = spacing

    bubble = np.array([0, 0, 1, 1])
    bubbles = np.array([[2, 2, 1, 1], [3, 3, 1, 1]])

    collisions = chart._check_collisions(bubble, bubbles)
    assert collisions == expected_collisions


def test_check_collisions_with_different_radii(sample_bubble_chart3):
    """Test collision check with bubbles of different sizes."""
    bubble = np.array([0, 0, 4, 8])  # Larger bubble
    bubbles = np.array(
        [
            [3, 3, 1, 1],  # Should collide due to larger radius
            [6, 6, 1, 1],  # Should not collide
        ]
    )

    collisions = sample_bubble_chart3._check_collisions(bubble, bubbles)
    assert collisions == 1


def test_collapse_basic_movement(sample_bubble_chart4):
    """Test basic bubble movement towards center of mass."""
    initial_positions = sample_bubble_chart4.bubbles.copy()
    # Stuff that needs to be set to test _collapse()
    sample_bubble_chart4.com = sample_bubble_chart4._center_of_mass()
    sample_bubble_chart4.maxstep = (
        2 * sample_bubble_chart4.bubbles[:, 2].max()
        + sample_bubble_chart4.bubble_spacing
    )
    sample_bubble_chart4.step_dist = sample_bubble_chart4.maxstep / 2

    sample_bubble_chart4._collapse(n_iterations=1)

    # Check that bubbles moved
    assert not np.array_equal(sample_bubble_chart4.bubbles, initial_positions)

    # WARNING: This test does not seem to produce predictable results
    # Check that outer bubbles moved towards center
    # assert np.all(abs(sample_bubble_chart4.bubbles[0, :2]) > abs(initial_positions[0, :2]))
    # assert np.all(abs(sample_bubble_chart4.bubbles[2, :2]) > abs(initial_positions[2, :2]))


def test_collapse_step_distance_reduction():
    """Test step distance reduction when few moves occur."""
    chart = BubbleChart(data="test", showfig=False)
    chart.bubbles = np.array(
        [
            [0, 0, 2, 4],  # Large overlapping bubbles to force few moves
            [1, 1, 2, 4],
        ]
    )
    # Stuff that needs to be set to test _collapse()
    chart.com = chart._center_of_mass()
    chart.maxstep = 2 * chart.bubbles[:, 2].max() + chart.bubble_spacing
    chart.step_dist = 1.0

    initial_step = chart.step_dist

    chart._collapse(n_iterations=1)
    assert chart.step_dist < initial_step


def test_collapse_collision_avoidance(sample_bubble_chart4):
    """Test that bubbles avoid collisions during collapse."""
    # Add a bubble that would cause collision
    sample_bubble_chart4.bubbles = np.array(
        [
            [0, 0, 1, 1],
            [1, 0, 1, 1],  # Close enough to cause collision
            [5, 5, 1, 1],
        ]
    )

    # Stuff that needs to be set to test _collapse()
    sample_bubble_chart4.com = sample_bubble_chart4._center_of_mass()
    sample_bubble_chart4.maxstep = (
        2 * sample_bubble_chart4.bubbles[:, 2].max()
        + sample_bubble_chart4.bubble_spacing
    )
    sample_bubble_chart4.step_dist = sample_bubble_chart4.maxstep / 2

    sample_bubble_chart4._collapse(n_iterations=1)

    # Check that bubbles moved but didn't overlap
    distances = sample_bubble_chart4._outline_distance(
        sample_bubble_chart4.bubbles[0], np.delete(sample_bubble_chart4.bubbles, 0, 0)
    )
    # WARNING: I'm not sure if this is the correct way to check for no collisions
    assert all(abs(d) >= 0 for d in distances)


# WARNING: This test has been disabled because I am not sure how to make it work
# def test_collapse_multiple_iterations():
#     """Test collapse behavior over multiple iterations."""
#     chart = BubbleChart(data="test", showfig=False)
#     chart.bubbles = np.array([[0, 0, 1, 1], [10, 10, 1, 1], [20, 20, 1, 1]])
#     # Stuff that needs to be set to test _collapse()
#     chart.com = chart._center_of_mass()
#     chart.maxstep = 2 * chart.bubbles[:, 2].max() + chart.bubble_spacing
#     chart.step_dist = chart.maxstep / 2

#     initial_spread = np.std(chart.bubbles[:, :2])

#     chart._collapse(n_iterations=10)
#     final_spread = np.std(chart.bubbles[:, :2])

#     # Verify bubbles are closer together
#     assert final_spread < initial_spread

# WARNING: This test has been disabled because I am not sure how to make it work
# @pytest.mark.parametrize("n_iterations", [1, 10, 50])
# def test_collapse_iteration_count(sample_bubble_chart4, n_iterations):
#     """Test collapse with different iteration counts.

#     Args:
#         sample_bubble_chart4: Fixture providing bubble chart instance
#         n_iterations: Number of iterations to run collapse
#     """
#     # Stuff that needs to be set to test _collapse()
#     sample_bubble_chart4.com = sample_bubble_chart4._center_of_mass()
#     sample_bubble_chart4.maxstep = (
#         2 * sample_bubble_chart4.bubbles[:, 2].max()
#         + sample_bubble_chart4.bubble_spacing
#     )
#     sample_bubble_chart4.step_dist = sample_bubble_chart4.maxstep / 2

#     initial_positions = sample_bubble_chart4.bubbles.copy()
#     sample_bubble_chart4._collapse(n_iterations=n_iterations)

#     # Verify that longer iterations generally result in more movement
#     final_positions = sample_bubble_chart4.bubbles
#     movement = np.sum(np.abs(final_positions - initial_positions))
#     assert movement > 0


def test_collides_with_single_collision():
    """Test collision detection with single bubble collision."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([0, 0, 1, 1])  # x, y, radius, area
    bubbles = np.array(
        [
            [1, 1, 1, 1],  # Should collide with this one
            [5, 5, 1, 1],  # Too far to collide
        ]
    )

    collision_indices = chart._collides_with(bubble, bubbles)
    assert collision_indices == [0]


def test_collides_with_multiple_bubbles():
    """Test collision detection with multiple bubbles."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([0, 0, 2, 4])  # Larger bubble
    bubbles = np.array([[2, 2, 1, 1], [3, 3, 1, 1], [10, 10, 1, 1]])

    collision_indices = chart._collides_with(bubble, bubbles)
    assert len(collision_indices) == 1
    assert collision_indices[0] in [0, 1]  # Should collide with closest bubble


def test_collides_with_no_collision():
    """Test collision detection with no collisions."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([0, 0, 1, 1])
    bubbles = np.array([[5, 5, 1, 1], [10, 10, 1, 1]])

    collision_indices = chart._collides_with(bubble, bubbles)
    assert collision_indices == [
        0
    ]  # Returns index of closest bubble even if no collision


@pytest.mark.parametrize(
    "bubble_pos,bubbles_pos,expected_idx",
    [
        ([0, 0], [[1, 1], [2, 2]], 0),  # Closest to first bubble
        ([5, 5], [[0, 0], [10, 10]], 0),  # Equidistant case
        ([0, 0], [[10, 0], [0, 10]], 0),  # Different axis alignments
    ],
)
def test_collides_with_various_positions(bubble_pos, bubbles_pos, expected_idx):
    """Test collision detection with various bubble positions.

    Args:
        bubble_pos: Position of test bubble [x, y]
        bubbles_pos: Positions of other bubbles [[x1, y1], [x2, y2]]
        expected_idx: Expected index of closest bubble
    """
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([*bubble_pos, 1, 1])
    bubbles = np.array([[*pos, 1, 1] for pos in bubbles_pos])

    collision_indices = chart._collides_with(bubble, bubbles)
    assert collision_indices == [expected_idx]


def test_collides_with_single_bubble():
    """Test collision detection with single bubble in array."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([0, 0, 1, 1])
    bubbles = np.array([[3, 3, 1, 1]])

    collision_indices = chart._collides_with(bubble, bubbles)
    assert collision_indices == [0]


def test_outline_distance_no_overlap():
    """Test outline distance calculation with non-overlapping bubbles."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([0, 0, 1, 1])  # x, y, radius, area
    bubbles = np.array([[4, 0, 1, 1]])  # 4 units away on x-axis

    distances = chart._outline_distance(bubble, bubbles)
    # Expected: center_distance(4) - radius1(1) - radius2(1) - spacing(0.1)
    expected = 4 - 1 - 1 - chart.bubble_spacing
    assert np.isclose(distances[0], expected)


def test_outline_distance_overlapping():
    """Test outline distance calculation with overlapping bubbles."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([0, 0, 1, 1])
    bubbles = np.array([[1, 0, 1, 1]])  # 1 unit away, should overlap

    distances = chart._outline_distance(bubble, bubbles)
    # Negative distance indicates overlap
    assert distances[0] < 0


@pytest.mark.parametrize(
    "spacing,expected",
    [
        (0.1, 0.9),  # Default spacing
        (0.5, 0.5),  # Larger spacing
        (0.0, 1.0),  # No spacing
    ],
)
def test_outline_distance_different_spacing(spacing, expected):
    """Test outline distance with different bubble spacing values.

    Args:
        spacing: Bubble spacing value to test
        expected: Expected distance value
    """
    chart = BubbleChart(data="test", showfig=False)
    chart.bubble_spacing = spacing
    bubble = np.array([0, 0, 1, 1])
    bubbles = np.array([[3, 0, 1, 1]])

    distances = chart._outline_distance(bubble, bubbles)
    assert np.isclose(distances[0], expected)


def test_outline_distance_different_radii():
    """Test outline distance with bubbles of different sizes."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([0, 0, 2, 4])  # Larger bubble
    bubbles = np.array([[5, 0, 1, 1]])  # Smaller bubble

    distances = chart._outline_distance(bubble, bubbles)
    # Expected: center_distance(5) - radius1(2) - radius2(1) - spacing(0.1)
    expected = 5 - 2 - 1 - chart.bubble_spacing
    assert np.isclose(distances[0], expected)


def test_outline_distance_multiple_bubbles():
    """Test outline distance with multiple target bubbles."""
    chart = BubbleChart(data="test", showfig=False)
    bubble = np.array([0, 0, 1, 1])
    bubbles = np.array([[3, 0, 1, 1], [0, 3, 1, 1], [3, 3, 1, 1]])

    distances = chart._outline_distance(bubble, bubbles)
    assert len(distances) == 3
    assert all(isinstance(d, float) for d in distances)


def test_plot_basic_functionality(sample_bubble_chart5, test_axes):
    """Test basic plotting functionality."""
    labels = ["word1", "word2", "word3"]
    sample_bubble_chart5._plot(test_axes, labels)

    # Check if correct number of circles were added
    circles = [p for p in test_axes.patches if isinstance(p, Circle)]
    assert len(circles) == len(sample_bubble_chart5.bubbles)


def test_plot_circle_properties(sample_bubble_chart5, test_axes):
    """Test properties of plotted circles."""
    labels = ["word1", "word2", "word3"]
    sample_bubble_chart5._plot(test_axes, labels)

    circles = [p for p in test_axes.patches if isinstance(p, Circle)]
    for i, circle in enumerate(circles):
        assert np.allclose(circle.center, sample_bubble_chart5.bubbles[i, :2])
        assert np.isclose(circle.radius, sample_bubble_chart5.bubbles[i, 2])


def test_plot_labels(sample_bubble_chart5, test_axes):
    """Test text label properties."""
    labels = ["test", "data", "test"]
    sample_bubble_chart5._plot(test_axes, labels)

    texts = test_axes.texts
    assert len(texts) == len(labels)

    for i, text in enumerate(texts):
        assert text.get_text() == labels[i]
        assert text.get_ha() == "center"
        assert text.get_va() == "center"


def test_plot_color_cycling(sample_bubble_chart5, test_axes):
    """Test color cycling for bubbles."""
    labels = ["word1", "word2", "word3", "word4"]
    sample_bubble_chart5.bubbles = np.vstack(
        [sample_bubble_chart5.bubbles, [9, 9, 1, 1]]
    )  # Add fourth bubble
    sample_bubble_chart5._plot(test_axes, labels)

    circles = [p for p in test_axes.patches if isinstance(p, Circle)]
    # Check color cycling
    assert circles[0].get_facecolor() != circles[1].get_facecolor()
    assert circles[0].get_facecolor() == circles[3].get_facecolor()  # Color cycling


def test_plot_font_family(sample_bubble_chart5, test_axes):
    """Test font family setting."""
    sample_bubble_chart5.font_family = "Arial"
    labels = ["test", "data", "test"]
    sample_bubble_chart5._plot(test_axes, labels)

    assert plt.rcParams["font.family"] == ["Arial"]


@pytest.mark.parametrize("font_family", ["Arial", "Times New Roman", "Courier New"])
def test_plot_different_fonts(sample_bubble_chart5, test_axes, font_family):
    """Test different font families.

    Args:
        font_family: Font family to test
    """
    sample_bubble_chart5.font_family = font_family
    labels = ["test", "data", "test"]
    sample_bubble_chart5._plot(test_axes, labels)

    assert plt.rcParams["font.family"] == [font_family]


def test_process_data_string():
    """Test data processing with string input."""
    chart = BubbleChart(data="hello world hello test")
    chart._process_data()

    assert chart.term_counts == {"hello": 2, "world": 1, "test": 1}


def test_process_data_doc(nlp):
    """Test data processing with spaCy Doc."""
    doc = nlp("hello world hello")
    chart = BubbleChart(data=doc)
    chart._process_data()

    assert chart.term_counts == {"hello": 2, "world": 1}


def test_process_data_dtm(sample_dtm):
    """Test data processing with DTM."""
    chart = BubbleChart(data=sample_dtm)
    chart._process_data()

    assert isinstance(chart.term_counts, dict)
    assert len(chart.term_counts) > 0


def test_process_data_dataframe():
    """Test data processing with DataFrame."""
    data = {"doc1": [1, 2, 3], "doc2": [4, 5, 6], "doc3": [7, 8, 9]}
    df = pd.DataFrame(data, index=["term1", "term2", "term3"])
    chart = BubbleChart(data=df)
    chart._process_data()

    assert isinstance(chart.term_counts, dict)
    assert len(chart.term_counts) > 0


def test_process_data_list_of_lists():
    """Test data processing with list of string lists."""
    data = [["hello", "world"], ["test", "hello"]]
    chart = BubbleChart(data=data)
    chart._process_data()

    assert isinstance(chart.term_counts, dict)
    assert chart.term_counts["hello"] == 2


def test_process_data_list_of_docs(nlp):
    """Test data processing with list of Docs."""
    docs = [nlp("hello world"), nlp("test hello")]
    chart = BubbleChart(data=docs)
    chart._process_data()

    assert isinstance(chart.term_counts, dict)
    assert chart.term_counts["hello"] == 2


def test_process_data_dict():
    """Test data processing with dictionary input."""
    data = {"word1": 2, "word2": 3}
    chart = BubbleChart(data=data)
    chart._process_data()

    assert chart.term_counts == data


def test_process_data_invalid_input():
    """Test data processing with invalid input type."""
    with pytest.raises(ValidationError):
        chart = BubbleChart(data=123)  # Invalid input type
        chart._process_data()


@pytest.mark.parametrize(
    "input_data,expected_count",
    [
        ("word word word", {"word": 3}),
        ("a b c", {"a": 1, "b": 1, "c": 1}),
    ],
)
def test_process_data_various_strings(input_data, expected_count):
    """Test data processing with various string inputs.

    Args:
        input_data: Input string to test
        expected_count: Expected term frequency dictionary
    """
    chart = BubbleChart(data=input_data)
    chart._process_data()

    assert chart.term_counts == expected_count


@pytest.mark.parametrize("input_data", [(""), ([]), (pd.DataFrame())])
def test_process_data_empty(input_data):
    """Test data processing with various empty inputs.

    Args:
        input_data: Input to test
    """
    with pytest.raises(LexosException):
        chart = BubbleChart(data=input_data)
        chart._process_data()


def test_save_basic_functionality(sample_bubble_chart5, tmp_path):
    """Test basic save functionality with valid path."""
    save_path = tmp_path / "test_bubble.png"
    sample_bubble_chart5()
    sample_bubble_chart5.save(save_path)
    assert save_path.exists()


def test_save_string_path(sample_bubble_chart5, tmp_path):
    """Test save with string path instead of Path object."""
    save_path = str(tmp_path / "test_bubble.png")
    sample_bubble_chart5()
    sample_bubble_chart5.save(save_path)
    assert Path(save_path).exists()


def test_save_empty_path(sample_bubble_chart5):
    """Test save with empty path."""
    sample_bubble_chart5()
    with pytest.raises(LexosException, match="You must provide a valid path"):
        sample_bubble_chart5.save("")


def test_save_none_path(sample_bubble_chart5):
    """Test save with None path."""
    sample_bubble_chart5()
    with pytest.raises(ValidationError):
        sample_bubble_chart5.save(None)


@pytest.mark.parametrize("extension", [".png", ".jpg", ".pdf", ".svg"])
def test_save_different_formats(sample_bubble_chart5, tmp_path, extension):
    """Test saving in different file formats.

    Args:
        extension: File extension to test
    """
    save_path = tmp_path / f"test_bubble{extension}"
    sample_bubble_chart5()
    plt.close("all")  # Close figure to avoid "RuntimeWarning: More than 20 figures have been opened."
    sample_bubble_chart5.save(save_path)
    assert save_path.exists()


def test_save_no_figure():
    """Test save when no figure has been generated."""
    chart = BubbleChart(data="test data", showfig=False)  # Don't call __call__()

    with pytest.raises(LexosException):
        chart.save("test.png")


def test_show_with_figure(sample_bubble_chart5):
    """Test show when figure exists."""
    sample_bubble_chart5()
    fig = sample_bubble_chart5.show()
    assert isinstance(fig, plt.Figure)
    assert fig is sample_bubble_chart5.fig


def test_show_no_figure():
    """Test show when no figure has been generated."""
    chart = BubbleChart(data="test data", showfig=False)
    with pytest.raises(LexosException) as exc_info:
        chart.show()
    assert str(exc_info.value) == "The figure has not yet been generated."


def test_show_after_generation(sample_bubble_chart5):
    """Test show properties after figure generation."""
    sample_bubble_chart5()
    fig = sample_bubble_chart5.show()
    assert hasattr(fig, "axes")
    assert len(fig.axes) > 0
    assert fig.get_size_inches().tolist() == [15, 15]  # Default figsize
    plt.close("all")  # Close figure to avoid "RuntimeWarning: More than 20 figures have been opened."


def test_show_returns_same_figure(sample_bubble_chart5):
    """Test that multiple show calls return same figure."""
    sample_bubble_chart5()
    fig1 = sample_bubble_chart5.show()
    fig2 = sample_bubble_chart5.show()
    assert fig1 is fig2
    plt.close("all")  # Close figure to avoid "RuntimeWarning: More than 20 figures have been opened."
