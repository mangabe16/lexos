"""test_plotly_dendrogram.py.

Last Update: February 27, 2025
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest
import spacy
from plotly.graph_objs import Figure

from lexos.cluster import PlotlyDendrogram
from lexos.dtm import DTM
from lexos.exceptions import LexosException

nlp = spacy.load("en_core_web_sm")

# Fixtures

@pytest.fixture
def sample_figure():
    """Create sample plotly figure for testing.

    Returns:
        go.Figure: Basic plotly figure with sample data
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
    fig.layout.xaxis.ticktext = ["Label1", "Label2", "Label3"]
    return fig

@pytest.fixture
def sample_dtm():
    """Create sample DTM for testing.

    Returns:
        DTM: Sample DTM instance with test data
    """
    dtm = DTM()
    dtm(
        docs=[nlp("kitten alert"), nlp("term1"), nlp("Term3"), nlp("10term"), nlp("2term")],
        labels=["Doc1", "doc2", "doc3", "doc4", "doc5"],
    )
    return dtm

@pytest.fixture
def basic_dendrogram():
    """Create basic PlotlyDendrogram instance.

    Returns:
        PlotlyDendrogram: Default configured dendrogram
    """
    return PlotlyDendrogram()

# Tests

def test_dendrogram_init(basic_dendrogram):
    """Test PlotlyDendrogram initialization with default values."""
    assert basic_dendrogram.metric == "euclidean"
    assert basic_dendrogram.method == "average"
    assert basic_dendrogram.orientation == "bottom"
    assert basic_dendrogram.figsize == (10, 10)
    assert basic_dendrogram.showfig is False

def test_dendrogram_with_dtm(sample_dtm):
    """Test PlotlyDendrogram creation with DTM input."""
    dendrogram = PlotlyDendrogram(dtm=sample_dtm, labels=sample_dtm.labels)
    assert dendrogram.dtm is not None
    assert dendrogram.labels == ['Doc1', 'doc2', 'doc3', 'doc4', 'doc5']

def test_dendrogram_with_dataframe():
    """Test PlotlyDendrogram creation with DataFrame input."""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })
    dendrogram = PlotlyDendrogram(dtm=df)
    assert dendrogram.dtm is not None

def test_dendrogram_custom_config():
    """Test PlotlyDendrogram with custom configuration."""
    custom_config = {
        'displaylogo': True,
        'scrollZoom': False
    }
    dendrogram = PlotlyDendrogram(config=custom_config)
    assert dendrogram.config['displaylogo'] is True
    assert dendrogram.config['scrollZoom'] is False

@pytest.mark.parametrize("orientation,tickangle", [
    ("bottom", 45),
    ("left", 90)
])
def test_dendrogram_orientation_tickangle(orientation, tickangle, sample_dtm):
    """Test different orientations and tick angles.

    Args:
        orientation: Dendrogram orientation
        tickangle: Angle for tick labels
        sample_dtm: Sample DTM fixture
    """
    dendrogram = PlotlyDendrogram(
        dtm=sample_dtm,
        orientation=orientation,
        x_tickangle=tickangle,
        y_tickangle=tickangle
    )
    dendrogram()
    assert isinstance(dendrogram.fig, Figure)

def test_dendrogram_custom_layout(sample_dtm):
    """Test PlotlyDendrogram with custom layout."""
    custom_layout = {
        'title': 'Test Dendrogram',
        'width': 800,
        'height': 600
    }
    dendrogram = PlotlyDendrogram(
        dtm=sample_dtm,
        layout=custom_layout
    )
    dendrogram()
    assert dendrogram.layout == custom_layout

def test_dendrogram_colorscale(sample_dtm):
    """Test PlotlyDendrogram with custom colorscale."""
    colorscale = [[0, 'red'], [1, 'blue']]
    dendrogram = PlotlyDendrogram(
        dtm=sample_dtm,
        colorscale=colorscale
    )
    dendrogram()
    assert dendrogram.colorscale == colorscale

def test_dendrogram_validation():
    """Test PlotlyDendrogram input validation."""
    with pytest.raises(LexosException):
        dendrogram = PlotlyDendrogram()
        dendrogram()  # Should raise exception for missing DTM

def test_dendrogram_show_behavior(sample_dtm):
    """Test PlotlyDendrogram show behavior."""
    # Test with showfig=False
    dendrogram_hidden = PlotlyDendrogram(
        dtm=sample_dtm,
        showfig=False
    )
    dendrogram_hidden()
    assert isinstance(dendrogram_hidden.fig, Figure)

    # Test with showfig=True
    dendrogram_shown = PlotlyDendrogram(
        dtm=sample_dtm,
        showfig=True
    )
    dendrogram_shown()
    assert isinstance(dendrogram_shown.fig, Figure)

def test_call_basic(sample_dtm):
    """Test basic dendrogram creation."""
    dendrogram = PlotlyDendrogram()
    dendrogram(dtm=sample_dtm)

    assert isinstance(dendrogram.fig, Figure)
    assert dendrogram.labels == ['Doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    assert dendrogram.metric == "euclidean"
    assert dendrogram.method == "average"

def test_call_no_dtm():
    """Test error handling with no DTM."""
    dendrogram = PlotlyDendrogram()
    with pytest.raises(LexosException) as exc_info:
        dendrogram()
    assert "You must provide a document-term matrix" in str(exc_info.value)

def test_call_custom_labels():
    """Test dendrogram with custom labels."""
    data = np.array([[1, 2], [3, 4]])
    custom_labels = ['A', 'B']
    dendrogram = PlotlyDendrogram()
    dendrogram(dtm=data, labels=custom_labels)

    assert dendrogram.labels == custom_labels

def test_call_with_dataframe(sample_dtm):
    """Test dendrogram with pandas DataFrame."""
    df = pd.DataFrame({
        'doc1': [1, 2, 3],
        'doc2': [4, 5, 6]
    })
    df.index.name = "terms"
    df = df.T
    df.columns = ['term1', 'term2', 'term3']
    dendrogram = PlotlyDendrogram()
    dendrogram(dtm=df)

    assert isinstance(dendrogram.fig, Figure)
    assert len(dendrogram.labels) == 2

def test_call_distance_functions(sample_dtm):
    """Test custom distance metrics and linkage methods."""
    dendrogram = PlotlyDendrogram()
    dendrogram(
        dtm=sample_dtm,
        metric="cosine",
        method="complete"
    )

    assert dendrogram.metric == "cosine"
    assert dendrogram.method == "complete"

def test_call_layout_options(sample_dtm):
    """Test layout customization."""
    custom_layout = {"width": 800, "height": 600}
    dendrogram = PlotlyDendrogram()
    dendrogram(
        dtm=sample_dtm,
        layout=custom_layout,
        title="Test Dendrogram",
        x_tickangle=45,
        y_tickangle=45
    )

    assert dendrogram.layout == custom_layout
    assert dendrogram.title == "Test Dendrogram"
    assert dendrogram.x_tickangle == 45
    assert dendrogram.y_tickangle == 45

def test_call_show_behavior(sample_dtm):
    """Test show behavior control."""
    # Test with show=False
    dendrogram_hidden = PlotlyDendrogram()
    dendrogram_hidden(dtm=sample_dtm, showfig=False)
    assert isinstance(dendrogram_hidden.fig, Figure)

    # Test with show=True
    dendrogram_shown = PlotlyDendrogram()
    dendrogram_shown(dtm=sample_dtm, showfig=True)
    assert isinstance(dendrogram_shown.fig, Figure)

@pytest.mark.parametrize("orientation", ["bottom", "left"])
def test_call_orientations(sample_dtm, orientation):
    """Test different dendrogram orientations.

    Args:
        orientation: Dendrogram orientation to test
        sample_dtm: Sample DTM fixture
    """
    dendrogram = PlotlyDendrogram()
    dendrogram(dtm=sample_dtm, orientation=orientation)

    assert dendrogram.orientation == orientation
    assert isinstance(dendrogram.fig, Figure)

def test_get_valid_matrix_dtm(sample_dtm):
    """Test matrix conversion from DTM."""
    dendrogram = PlotlyDendrogram(dtm=sample_dtm)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (5, 6)  # Transposed from 6x5

def test_get_valid_matrix_dataframe():
    """Test matrix handling with pandas DataFrame."""
    df = pd.DataFrame({
        'doc1': [1, 2, 3],
        'doc2': [4, 5, 6]
    })
    df.index.name = "terms"
    df = df.T
    df.columns = ['term1', 'term2', 'term3']
    dendrogram = PlotlyDendrogram(dtm=df)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (2, 3)

def test_get_valid_matrix_numpy():
    """Test matrix handling with numpy array."""
    array = np.array([[1, 2], [3, 4], [5, 6]])
    dendrogram = PlotlyDendrogram(dtm=array)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 2)

def test_get_valid_matrix_list():
    """Test matrix handling with list input."""
    list_data = [[1, 2], [3, 4], [5, 6]]
    dendrogram = PlotlyDendrogram(dtm=list_data)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, list)
    assert len(matrix) == 3

def test_get_valid_matrix_single_document():
    """Test error handling for single document matrix."""
    single_doc = np.array([[1, 2, 3]])
    dendrogram = PlotlyDendrogram(dtm=single_doc)

    with pytest.raises(LexosException) as exc_info:
        dendrogram._get_valid_matrix()
    assert "must have more than one document" in str(exc_info.value)

def test_get_valid_matrix_single_document_list():
    """Test error handling for single document list."""
    single_doc_list = [[1, 2, 3]]
    dendrogram = PlotlyDendrogram(dtm=single_doc_list)

    with pytest.raises(LexosException) as exc_info:
        dendrogram._get_valid_matrix()
    assert "must have more than one document" in str(exc_info.value)

def test_get_valid_matrix_empty():
    """Test error handling for empty matrix."""
    empty_array = np.array([[]])
    dendrogram = PlotlyDendrogram(dtm=empty_array)

    with pytest.raises(LexosException) as exc_info:
        dendrogram._get_valid_matrix()
    assert "must have more than one document" in str(exc_info.value)

