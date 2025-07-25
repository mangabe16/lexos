"""test_clustermap.py.

Last Update: February 27, 2025
"""

import os

import matplotlib as mpl  # added
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import spacy
from scipy.cluster.hierarchy import linkage
import seaborn as sns

from lexos.cluster import ClusterMap
from lexos.dtm import DTM
from lexos.exceptions import LexosException


nlp = spacy.load("en_core_web_sm")
mpl.use("Agg")  # added


# Fixtures


@pytest.fixture
def sample_dtm():
    """Create sample DTM for testing.

    Returns:
        DTM: Sample DTM instance with test data
    """
    dtm = DTM()
    dtm(
        docs=[
            nlp("kitten alert"),
            nlp("term1"),
            nlp("Term3"),
            nlp("10term"),
            nlp("2term"),
        ],
        labels=["Doc1", "doc2", "doc3", "doc4", "doc5"],
    )
    return dtm


@pytest.fixture
def sample_dtm2():
    """Create sample DTM for testing.

    Returns:
        DTM: Sample DTM instance with test data
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    terms = ["term1", "term2", "term3"]
    docs = ["doc1", "doc2", "doc3"]
    return DTM(matrix=data, terms=terms, labels=docs)


@pytest.fixture
def sample_dataframe():
    """Create sample pandas DataFrame for testing.

    Args:
        None

    Returns:
        pd.DataFrame: Sample DataFrame with test data
    """
    return pd.DataFrame({"id": ["1", "2", "3"], "col1": [1, 2, 3], "col2": [4, 5, 6]})


@pytest.fixture
def sample_linkage_data():
    """Create sample data for linkage computation.

    Returns:
        np.ndarray: Sample data array
    """
    return np.array([[1, 2], [3, 4], [5, 6]])


@pytest.fixture
def valid_linkage_matrix(sample_linkage_data):
    """Create valid linkage matrix.

    Args:
        sample_linkage_data: Sample data fixture

    Returns:
        np.ndarray: Valid linkage matrix
    """
    return linkage(sample_linkage_data, method="average", metric="euclidean")


@pytest.fixture
def sample_clustermap(sample_dtm):
    """Create a sample clustermap with data.

    Returns:
        ClusterMap: Configured clustermap instance with test data
    """
    clustermap = ClusterMap(dtm=sample_dtm)
    clustermap()  # Create the figure
    return clustermap


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs.

    Args:
        tmp_path: pytest fixture providing temporary directory

    Returns:
        Path: Temporary directory path
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_data():
    """Create sample data for testing.

    Returns:
        tuple: (data array, row linkage, column linkage)
    """
    # Create sample data
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Compute sample linkage matrices
    row_link = linkage(data, method="average", metric="euclidean")
    col_link = linkage(data.T, method="average", metric="euclidean")

    return data, row_link, col_link


# Tests


def test_clustermap_init_default():
    """Test ClusterMap initialization with default values."""
    clustermap = ClusterMap()

    assert clustermap.dtm is None
    assert clustermap.labels is None
    assert clustermap.metric == "euclidean"
    assert clustermap.method == "average"
    assert clustermap.hide_upper is False
    assert clustermap.hide_side is False
    assert clustermap.figsize == (8, 8)
    assert clustermap.cmap == "vlag"
    assert clustermap.linewidths == 0.75


def test_clustermap_init_with_dtm(sample_dtm):
    """Test initialization with DTM input."""
    clustermap = ClusterMap(dtm=sample_dtm)

    assert isinstance(clustermap.dtm, DTM)
    assert clustermap.labels is None  # Labels are set during __call__


def test_clustermap_init_with_custom_values():
    """Test initialization with custom parameter values."""
    custom_tree_kws = {"linewidth": 2.0}
    clustermap = ClusterMap(
        metric="cosine",
        method="complete",
        hide_upper=True,
        hide_side=True,
        title="Test ClusterMap",
        figsize=(12, 12),
        row_cluster=False,
        col_cluster=False,
        tree_kws=custom_tree_kws,
        cmap="RdBu",
    )

    assert clustermap.metric == "cosine"
    assert clustermap.method == "complete"
    assert clustermap.hide_upper is True
    assert clustermap.hide_side is True
    assert clustermap.title == "Test ClusterMap"
    assert clustermap.figsize == (12, 12)
    assert clustermap.row_cluster is False
    assert clustermap.col_cluster is False
    assert clustermap.tree_kws == custom_tree_kws
    assert clustermap.cmap == "RdBu"


def test_clustermap_color_parameters():
    """Test color-related parameter initialization."""
    custom_colors = ["red", "blue", "green"]
    clustermap = ClusterMap(
        row_colors=custom_colors,
        col_colors=custom_colors,
        colors_ratio=0.05,
        cmap="viridis",
    )

    assert clustermap.row_colors == custom_colors
    assert clustermap.col_colors == custom_colors
    assert clustermap.colors_ratio == 0.05
    assert clustermap.cmap == "viridis"


def test_clustermap_dendrogram_parameters():
    """Test dendrogram-related parameter initialization."""
    clustermap = ClusterMap(dendrogram_ratio=(0.2, 0.3))

    assert clustermap.dendrogram_ratio == (0.2, 0.3)


@pytest.mark.parametrize("standard_scale,z_score", [(0, 0), (1, 1), (None, 2)])
def test_clustermap_scaling_parameters(standard_scale, z_score):
    """Test scaling parameter initialization.

    Args:
        standard_scale: Standard scale parameter value
        z_score: Z-score parameter value
    """
    clustermap = ClusterMap(standard_scale=standard_scale, z_score=z_score)

    assert clustermap.standard_scale == standard_scale
    assert clustermap.z_score == z_score


def test_call_basic(sample_dtm):
    """Test basic clustermap creation."""
    clustermap = ClusterMap()
    clustermap(dtm=sample_dtm)

    assert isinstance(clustermap.fig, plt.Figure)
    assert len(clustermap.labels) == 5
    assert clustermap.method == "average"
    assert clustermap.metric == "euclidean"
    plt.close(clustermap.fig)


def test_call_no_dtm():
    """Test error handling with no DTM."""
    clustermap = ClusterMap()
    with pytest.raises(LexosException) as exc_info:
        clustermap()
    assert "You must provide a document-term matrix" in str(exc_info.value)


def test_call_with_custom_labels(sample_dtm):
    """Test clustermap with custom labels."""
    custom_labels = ["A", "B", "C", "D", "E"]
    clustermap = ClusterMap()
    clustermap(dtm=sample_dtm, labels=custom_labels)

    assert clustermap.labels == custom_labels


def test_call_row_color_len_match(sample_dtm):
    """Test clustermap with custom labels."""
    row_colors = ["red", "blue", "green"]
    custom_labels = ["A", "B", "C", "D", "E"]

    clustermap = ClusterMap()

    with pytest.raises(LexosException) as exc_info:
        clustermap(
            dtm=sample_dtm,
            row_colors=row_colors,
            colors_ratio=0.05,
            labels=custom_labels,
        )
    assert (
        "The length of `row_colors` must have be greater than the number of labels."
        in str(exc_info.value)
    )


def test_call_default_color_check(sample_dtm):
    """Test default color palette option."""
    default_colors = "default"

    clustermap = ClusterMap()
    clustermap(
        dtm=sample_dtm,
        row_colors=default_colors,
        col_colors=default_colors,
        colors_ratio=0.05,
    )
    clustermap()
    expected_palette = sns.husl_palette(8, s=0.45)
    actual_col_colors, actual_row_colors = clustermap._get_colors()

    assert np.allclose(actual_col_colors, expected_palette)
    assert np.allclose(actual_row_colors, expected_palette)


def test_call_invalid_col_palette_name(sample_dtm):
    """Test clustermap with invalid custom column palette."""
    col_colors = "this_palette_does_not_exist"
    clustermap = ClusterMap()

    with pytest.raises(LexosException) as exc_info:
        clustermap(dtm=sample_dtm, col_colors=col_colors, colors_ratio=0.05)
    assert "Invalid column palette." in str(exc_info.value)


def test_call_invalid_row_palette_name(sample_dtm):
    """Test clustermap with invalid custom row palette."""
    row_colors = "this_palette_does_not_exist"
    clustermap = ClusterMap()

    with pytest.raises(LexosException) as exc_info:
        clustermap(dtm=sample_dtm, row_colors=row_colors, colors_ratio=0.05)
    assert "Invalid row palette." in str(exc_info.value)


def test_call_with_parameters(sample_dtm):
    """Test clustermap with various parameters."""
    clustermap = ClusterMap()
    clustermap(
        dtm=sample_dtm,
        metric="cosine",
        method="complete",
        figsize=(12, 12),
        cmap="RdBu",
        z_score=1,
        title="Test ClusterMap",
    )

    assert clustermap.metric == "cosine"
    assert clustermap.method == "complete"
    assert clustermap.figsize == (12, 12)
    assert clustermap.cmap == "RdBu"
    assert clustermap.z_score == 1
    assert clustermap.title == "Test ClusterMap"


def test_call_color_options(sample_dtm):
    """Test color-related options."""
    row_colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    col_colors = ["red", "blue", "green", "yellow", "purple", "orange"]

    clustermap = ClusterMap()
    clustermap(
        dtm=sample_dtm, row_colors=row_colors, col_colors=col_colors, colors_ratio=0.05
    )

    assert sorted(clustermap.row_colors) == sorted(row_colors)
    assert sorted(clustermap.col_colors) == sorted(col_colors)
    assert clustermap.colors_ratio == 0.05


def test_call_show_behavior(sample_dtm):
    """Test show/hide behavior."""

    # Test with show=False
    clustermap_hidden = ClusterMap()
    clustermap_hidden(dtm=sample_dtm, showfig=False)
    assert clustermap_hidden.showfig is False
    assert isinstance(clustermap_hidden.fig, plt.Figure)

    # Test with show=True
    clustermap_shown = ClusterMap()
    clustermap_shown(dtm=sample_dtm, showfig=True)
    assert clustermap_shown.showfig is True
    assert isinstance(clustermap_shown.fig, plt.Figure)


@pytest.mark.parametrize("ratio", [0.1, (0.1, 0.2), (0.2, 0.3)])
def test_call_dendrogram_ratio(ratio, sample_dtm):
    """Test different dendrogram ratios.

    Args:
        ratio: Dendrogram ratio to test
    """
    clustermap = ClusterMap()
    clustermap(dtm=sample_dtm, dendrogram_ratio=ratio)

    assert clustermap.dendrogram_ratio == ratio


def test_clustermap_with_precomputed_linkage(sample_data):
    """Test clustermap creation with precomputed linkage matrices."""
    data, row_linkage, col_linkage = sample_data

    clustermap = ClusterMap()
    clustermap(dtm=data, row_linkage=row_linkage, col_linkage=col_linkage)

    assert clustermap.row_linkage is not None
    assert clustermap.col_linkage is not None
    assert isinstance(clustermap.fig, plt.Figure)
    plt.close(clustermap.fig)


@pytest.mark.filterwarnings(
    "ignore:Attempting to set identical low and high ylims:UserWarning"
)
def test_clustermap_row_linkage_only(sample_data):
    """Test clustermap with only row linkage specified."""
    data, row_linkage, _ = sample_data

    clustermap = ClusterMap()
    clustermap(dtm=data, row_linkage=row_linkage)

    assert clustermap.row_linkage is not None
    assert clustermap.col_linkage is None
    assert isinstance(clustermap.fig, plt.Figure)
    plt.close(clustermap.fig)


def test_clustermap_col_linkage_only(sample_data):
    """Test clustermap with only column linkage specified."""
    data, _, col_linkage = sample_data

    clustermap = ClusterMap()
    clustermap(dtm=data, col_linkage=col_linkage)

    assert clustermap.row_linkage is None
    assert clustermap.col_linkage is not None
    assert isinstance(clustermap.fig, plt.Figure)
    plt.close(clustermap.fig)


def test_clustermap_linkage_shape_validation(sample_data):
    """Test validation of linkage matrix shapes."""
    data, _, _ = sample_data

    # Create invalid linkage matrices (wrong shape)
    invalid_linkage = np.array([[1, 2, 3, 4]])

    clustermap = ClusterMap()
    with pytest.raises(LexosException):
        clustermap(dtm=data, row_linkage=invalid_linkage)


def test_clustermap_different_linkage_methods(sample_data):
    """Test clustermap with linkage matrices using different methods."""
    data, _, _ = sample_data

    # Create linkage matrices with different methods
    row_linkage_single = linkage(data, method="single", metric="euclidean")
    col_linkage_complete = linkage(data.T, method="complete", metric="euclidean")

    clustermap = ClusterMap()
    clustermap(
        dtm=data, row_linkage=row_linkage_single, col_linkage=col_linkage_complete
    )

    assert isinstance(clustermap.fig, plt.Figure)
    plt.close(clustermap.fig)


@pytest.mark.filterwarnings(
    "ignore:Attempting to set identical low and high ylims:UserWarning"
)
@pytest.mark.parametrize("metric", ["euclidean", "correlation", "cosine"])
def test_clustermap_linkage_metrics(sample_data, metric):
    """Test clustermap with linkage matrices using different metrics.

    Args:
        sample_data: Fixture providing test data
        metric: Distance metric to use for linkage computation
    """
    data, _, _ = sample_data

    row_linkage = linkage(data, method="average", metric=metric)
    col_linkage = linkage(data.T, method="average", metric=metric)

    clustermap = ClusterMap()
    clustermap(dtm=data, row_linkage=row_linkage, col_linkage=col_linkage)

    assert isinstance(clustermap.fig, plt.Figure)
    plt.close(clustermap.fig)


def test_get_valid_matrix_dtm(sample_dtm):
    """Test matrix conversion from DTM."""
    dendrogram = ClusterMap(dtm=sample_dtm)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (5, 6)  # Transposed from 6x5


def test_get_valid_matrix_dataframe():
    """Test matrix handling with pandas DataFrame."""
    df = pd.DataFrame({"doc1": [1, 2, 3], "doc2": [4, 5, 6]})
    df.index.name = "terms"
    df = df.T
    df.columns = ["term1", "term2", "term3"]
    dendrogram = ClusterMap(dtm=df)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (2, 3)


def test_get_valid_matrix_numpy():
    """Test matrix handling with numpy array."""
    array = np.array([[1, 2], [3, 4], [5, 6]])
    dendrogram = ClusterMap(dtm=array)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 2)


def test_get_valid_matrix_list():
    """Test matrix handling with list input."""
    list_data = [[1, 2], [3, 4], [5, 6]]
    dendrogram = ClusterMap(dtm=list_data)
    matrix = dendrogram._get_valid_matrix()

    assert isinstance(matrix, list)
    assert len(matrix) == 3


def test_get_valid_matrix_single_document():
    """Test error handling for single document matrix."""
    single_doc = np.array([[1, 2, 3]])
    dendrogram = ClusterMap(dtm=single_doc)

    with pytest.raises(LexosException) as exc_info:
        dendrogram._get_valid_matrix()
    assert "must have more than one document" in str(exc_info.value)


def test_get_valid_matrix_single_document_list():
    """Test error handling for single document list."""
    single_doc_list = [[1, 2, 3]]
    dendrogram = ClusterMap(dtm=single_doc_list)

    with pytest.raises(LexosException) as exc_info:
        dendrogram._get_valid_matrix()
    assert "must have more than one document" in str(exc_info.value)


def test_get_valid_matrix_empty():
    """Test error handling for empty matrix."""
    empty_array = np.array([[]])
    dendrogram = ClusterMap(dtm=empty_array)

    with pytest.raises(LexosException) as exc_info:
        dendrogram._get_valid_matrix()
    assert "must have more than one document" in str(exc_info.value)


def test_set_labels_with_dtm(sample_dtm):
    """Test label setting with DTM input."""
    clustermap = ClusterMap(dtm=sample_dtm)
    clustermap._set_labels()

    assert clustermap.labels == ["Doc1", "doc2", "doc3", "doc4", "doc5"]


def test_set_labels_with_dataframe(sample_dataframe):
    """Test label setting with DataFrame input."""
    clustermap = ClusterMap(dtm=sample_dataframe)
    clustermap._set_labels()

    assert clustermap.labels == ["col1", "col2"]


def test_set_labels_with_array():
    """Test label setting with numpy array input."""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    clustermap = ClusterMap(dtm=data)
    clustermap._set_labels()

    assert clustermap.labels == ["Doc1", "Doc2", "Doc3"]


def test_set_labels_custom_labels():
    """Test setting custom labels."""
    data = np.array([[1, 2], [3, 4]])
    custom_labels = ["Label1", "Label2"]
    clustermap = ClusterMap(dtm=data, labels=custom_labels)
    clustermap._set_labels()

    assert clustermap.labels == custom_labels


def test_set_labels_existing_labels():
    """Test label setting when labels already exist."""
    data = np.array([[1, 2], [3, 4]])
    initial_labels = ["Initial1", "Initial2"]
    clustermap = ClusterMap(dtm=data, labels=initial_labels)

    # First set labels
    clustermap._set_labels()
    assert clustermap.labels == initial_labels

    # Try to set labels again
    clustermap._set_labels()
    assert clustermap.labels == initial_labels  # Should remain unchanged


@pytest.mark.parametrize(
    "size,expected_labels",
    [
        (2, ["Doc1", "Doc2"]),
        (3, ["Doc1", "Doc2", "Doc3"]),
        (4, ["Doc1", "Doc2", "Doc3", "Doc4"]),
    ],
)
def test_set_labels_different_sizes(size, expected_labels):
    """Test label setting with different matrix sizes.

    Args:
        size: Size of test matrix
        expected_labels: Expected generated labels
    """
    data = np.zeros((size, size))  # Create square matrix of given size
    clustermap = ClusterMap(dtm=data)
    clustermap._set_labels()

    assert clustermap.labels == expected_labels


def test_validate_valid_linkage_matrices(valid_linkage_matrix):
    """Test validation of valid linkage matrices."""
    clustermap = ClusterMap(
        row_linkage=valid_linkage_matrix, col_linkage=valid_linkage_matrix
    )

    # Should not raise any exceptions
    clustermap._validate_linkage_matrices()


def test_validate_invalid_row_linkage():
    """Test validation with invalid row linkage matrix."""
    invalid_linkage = np.array([[1, 2, 3]])  # Wrong shape
    clustermap = ClusterMap(row_linkage=invalid_linkage)

    with pytest.raises(LexosException) as exc_info:
        clustermap._validate_linkage_matrices()
    assert "Invalid `row_linkage` value" in str(exc_info.value)


def test_validate_invalid_col_linkage():
    """Test validation with invalid column linkage matrix."""
    invalid_linkage = np.array([[1, 2]])  # Wrong shape
    clustermap = ClusterMap(col_linkage=invalid_linkage)

    with pytest.raises(LexosException) as exc_info:
        clustermap._validate_linkage_matrices()
    assert "Invalid `col_linkage` value" in str(exc_info.value)


def test_validate_none_linkage():
    """Test validation when linkage matrices are None."""
    clustermap = ClusterMap()

    # Should not raise any exceptions
    clustermap._validate_linkage_matrices()


def test_validate_invalid_type():
    """Test validation with wrong data type."""
    invalid_linkage = "not a linkage matrix"
    clustermap = ClusterMap()
    clustermap.row_linkage = invalid_linkage

    with pytest.raises(LexosException) as exc_info:
        clustermap._validate_linkage_matrices()
    assert "Invalid `row_linkage` value" in str(exc_info.value)


@pytest.mark.parametrize("method", ["single", "complete", "average", "weighted"])
def test_validate_different_linkage_methods(sample_linkage_data, method):
    """Test validation with different linkage methods.

    Args:
        sample_linkage_data: Sample data fixture
        method: Linkage method to test
    """
    linkage_matrix = linkage(sample_linkage_data, method=method, metric="euclidean")
    clustermap = ClusterMap(row_linkage=linkage_matrix)

    # Should not raise any exceptions
    clustermap._validate_linkage_matrices()


def test_save_png(sample_clustermap, temp_output_dir):
    """Test saving clustermap as PNG."""
    output_path = temp_output_dir / "clustermap.png"
    sample_clustermap.save(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_pdf(sample_clustermap, temp_output_dir):
    """Test saving clustermap as PDF."""
    output_path = temp_output_dir / "clustermap.pdf"
    sample_clustermap.save(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_with_custom_dpi(sample_clustermap, temp_output_dir):
    """Test saving with custom DPI setting."""
    output_path = temp_output_dir / "clustermap_high_dpi.png"
    sample_clustermap.save(output_path, dpi=300)

    assert output_path.exists()
    # Higher DPI should result in larger file
    assert output_path.stat().st_size > 1000


def test_save_str_path(sample_clustermap, temp_output_dir):
    """Test saving using string path instead of Path object."""
    output_path = str(temp_output_dir / "clustermap_str.png")
    sample_clustermap.save(output_path)

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_save_no_figure():
    """Test save attempt without creating figure first."""
    clustermap = ClusterMap()  # No data, no figure created

    with pytest.raises(AttributeError):
        clustermap.save("test.png")


@pytest.mark.parametrize("fmt", ["png", "pdf", "svg", "jpg"])
def test_save_different_formats(sample_clustermap, temp_output_dir, fmt):
    """Test saving in different file formats.

    Args:
        sample_clustermap: Configured clustermap fixture
        temp_output_dir: Temporary directory fixture
        fmt: File format to test
    """
    output_path = temp_output_dir / f"clustermap.{fmt}"
    sample_clustermap.save(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_with_kwargs(sample_clustermap, temp_output_dir):
    """Test saving with additional matplotlib kwargs."""
    output_path = temp_output_dir / "clustermap_custom.png"
    sample_clustermap.save(output_path, dpi=300, bbox_inches="tight", transparent=True)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_show_without_calling_instance(sample_dtm):
    """Test show() when figure hasn't been created yet."""
    clustermap = ClusterMap(dtm=sample_dtm)
    with pytest.raises(LexosException) as exc_info:
        clustermap.show()
        plt.close(clustermap.fig)
    assert "You must call the instance before showing the figure" in str(exc_info.value)


def test_show_with_figure(sample_dtm):
    """Test show() with properly created figure."""
    # Create figure by calling instance
    clustermap = ClusterMap(dtm=sample_dtm)
    clustermap()

    # Test show method
    fig = clustermap.show()
    assert isinstance(fig, plt.Figure)
    assert fig is clustermap.fig
    plt.close(clustermap.fig)


def test_show_after_close(sample_dtm):
    """Test show() after closing figure."""
    clustermap = ClusterMap(dtm=sample_dtm)
    clustermap()

    # Close the figure
    plt.close(clustermap.fig)

    # Show should still work and return the figure
    fig = clustermap.show()
    assert isinstance(fig, plt.Figure)
