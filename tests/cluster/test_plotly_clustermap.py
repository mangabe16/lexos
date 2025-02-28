"""test_plotly_clustermap.py.

Last Update: February 27, 2025
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest
import spacy
from plotly.graph_objs import Figure

from lexos.cluster import PlotlyClustermap
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
def basic_clustermap():
    """Create basic PlotlyClustermap instance.

    Returns:
        PlotlyClustermap: Default configured clustermap
    """
    return PlotlyClustermap()

# Tests

def test_clustermap_init_default():
    """Test PlotlyClustermap initialization with default values."""
    clustermap = PlotlyClustermap()

    assert clustermap.dtm is None
    assert clustermap.labels is None
    assert clustermap.metric == "euclidean"
    assert clustermap.method == "average"
    assert clustermap.hide_upper is False
    assert clustermap.hide_side is False
    assert clustermap.width == 600
    assert clustermap.height == 600
    assert clustermap.showfig is False
    assert clustermap.colorscale == "Viridis"

def test_clustermap_init_with_dtm(sample_dtm):
    """Test initialization with DTM input."""
    clustermap = PlotlyClustermap(dtm=sample_dtm)

    assert clustermap.dtm is not None
    assert isinstance(clustermap.dtm, DTM)
    assert clustermap.labels is None  # Labels are set during __call__

def test_clustermap_init_with_custom_values():
    """Test initialization with custom parameter values."""
    custom_config = {
        "displaylogo": True,
        "scrollZoom": False
    }

    clustermap = PlotlyClustermap(
        metric="cosine",
        method="complete",
        hide_upper=True,
        hide_side=True,
        width=800,
        height=800,
        title="Test Clustermap",
        colorscale="Reds",
        config=custom_config
    )

    assert clustermap.metric == "cosine"
    assert clustermap.method == "complete"
    assert clustermap.hide_upper is True
    assert clustermap.hide_side is True
    assert clustermap.width == 800
    assert clustermap.height == 800
    assert clustermap.title == "Test Clustermap"
    assert clustermap.colorscale == "Reds"
    assert clustermap.config == custom_config

def test_clustermap_config_default():
    """Test default configuration settings."""
    clustermap = PlotlyClustermap()

    assert clustermap.config["displaylogo"] is False
    assert "toImage" in clustermap.config["modeBarButtonsToRemove"]
    assert "toggleSpikelines" in clustermap.config["modeBarButtonsToRemove"]
    assert clustermap.config["scrollZoom"] is True

def test_clustermap_layout_default():
    """Test default layout settings."""
    clustermap = PlotlyClustermap()

    assert isinstance(clustermap.layout, dict)
    assert len(clustermap.layout) == 0

@pytest.mark.parametrize("width,height", [
    (400, 400),
    (800, 600),
    (1000, 1000)
])
def test_clustermap_dimensions(width, height):
    """Test different width and height configurations.

    Args:
        width: Width of the clustermap
        height: Height of the clustermap
    """
    clustermap = PlotlyClustermap(width=width, height=height)

    assert clustermap.width == width
    assert clustermap.height == height

def test_call_basic(sample_dtm):
    """Test basic clustermap creation."""
    clustermap = PlotlyClustermap()
    clustermap(dtm=sample_dtm)

    assert isinstance(clustermap.fig, Figure)
    assert len(clustermap.labels) == 5
    assert not clustermap.hide_upper
    assert not clustermap.hide_side

def test_call_no_dtm():
    """Test error handling with no DTM."""
    clustermap = PlotlyClustermap()
    with pytest.raises(LexosException) as exc_info:
        clustermap()
    assert "You must provide a document-term matrix" in str(exc_info.value)

def test_call_with_custom_labels():
    """Test clustermap with custom labels."""
    data = np.array([[1, 2], [3, 4]])
    custom_labels = ['A', 'B']
    clustermap = PlotlyClustermap()
    clustermap(dtm=data, labels=custom_labels)

    assert clustermap.labels == custom_labels

def test_call_hide_dendrograms():
    """Test hiding upper and side dendrograms."""
    data = np.array([[1, 2], [3, 4]])
    clustermap = PlotlyClustermap()
    clustermap(dtm=data, hide_upper=True, hide_side=True)

    assert clustermap.hide_upper
    assert clustermap.hide_side

def test_call_custom_dimensions():
    """Test custom width and height settings."""
    data = np.array([[1, 2], [3, 4]])
    clustermap = PlotlyClustermap()
    clustermap(dtm=data, width=800, height=800)

    assert clustermap.width == 800
    assert clustermap.height == 800

def test_call_custom_colorscale():
    """Test custom colorscale setting."""
    data = np.array([[1, 2], [3, 4]])
    clustermap = PlotlyClustermap()
    clustermap(dtm=data, colorscale="Reds")

    assert clustermap.colorscale == "Reds"

@pytest.mark.parametrize("metric,method", [
    ("euclidean", "single"),
    ("cosine", "complete"),
    ("correlation", "average")
])
def test_call_distance_metrics(sample_dtm, metric, method):
    """Test different distance metrics and linkage methods.

    Args:
        sample_dtm: Sample DTM fixture
        metric: Distance metric to use
        method: Linkage method to use
    """
    clustermap = PlotlyClustermap()
    clustermap(dtm=sample_dtm, metric=metric, method=method)

    assert clustermap.metric == metric
    assert clustermap.method == method
    assert isinstance(clustermap.fig, Figure)

def test_call_with_title(sample_dtm):
    """Test clustermap with title."""
    title = "Test Clustermap"
    clustermap = PlotlyClustermap()
    clustermap(dtm=sample_dtm, title=title)

    assert clustermap.title == title
    assert isinstance(clustermap.fig, Figure)

def test_call_custom_config(sample_dtm):
    """Test custom configuration settings."""
    custom_config = {
        "displaylogo": True,
        "scrollZoom": False
    }
    clustermap = PlotlyClustermap()
    clustermap(dtm=sample_dtm, config=custom_config)

    # custom_config["responsive"] = True
    assert clustermap.config == custom_config

def test_get_valid_matrix_dtm(sample_dtm):
    """Test matrix conversion from DTM."""
    dendrogram = PlotlyClustermap(dtm=sample_dtm)
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
    dendrogram = PlotlyClustermap(dtm=df)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (2, 3)

def test_get_valid_matrix_numpy():
    """Test matrix handling with numpy array."""
    array = np.array([[1, 2], [3, 4], [5, 6]])
    dendrogram = PlotlyClustermap(dtm=array)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 2)

def test_get_valid_matrix_list():
    """Test matrix handling with list input."""
    list_data = [[1, 2], [3, 4], [5, 6]]
    dendrogram = PlotlyClustermap(dtm=list_data)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, list)
    assert len(matrix) == 3

def test_get_valid_matrix_single_document():
    """Test error handling for single document matrix."""
    single_doc = np.array([[1, 2, 3]])
    dendrogram = PlotlyClustermap(dtm=single_doc)

    with pytest.raises(LexosException) as exc_info:
        dendrogram._get_valid_matrix()
    assert "must have more than one document" in str(exc_info.value)

def test_get_valid_matrix_single_document_list():
    """Test error handling for single document list."""
    single_doc_list = [[1, 2, 3]]
    dendrogram = PlotlyClustermap(dtm=single_doc_list)

    with pytest.raises(LexosException) as exc_info:
        dendrogram._get_valid_matrix()
    assert "must have more than one document" in str(exc_info.value)

def test_get_valid_matrix_empty():
    """Test error handling for empty matrix."""
    empty_array = np.array([[]])
    dendrogram = PlotlyClustermap(dtm=empty_array)

    with pytest.raises(LexosException) as exc_info:
        dendrogram._get_valid_matrix()
    assert "must have more than one document" in str(exc_info.value)
