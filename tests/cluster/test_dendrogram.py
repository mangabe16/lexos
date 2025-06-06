"""test_dendrogram.py.

Last Update: February 27, 2025
"""

import os

import matplotlib as mpl  # added
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import spacy

from lexos.cluster import Dendrogram
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
    # Create sample data
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
def basic_dendrogram():
    """Create basic Dendrogram instance.

    Returns:
        Dendrogram: Default configured Dendrogram
    """
    return Dendrogram()


@pytest.fixture
def sample_dendrogram():
    """Create sample Dendrogram instance with test data.

    Returns:
        Dendrogram: Configured dendrogram instance with test plot
    """
    # Create sample data
    data = np.array([[1, 2], [3, 4], [5, 6]])
    dendrogram = Dendrogram(dtm=data)
    dendrogram()  # Generate the plot
    return dendrogram


# Tests


def test_dendrogram_init(basic_dendrogram):
    """Test Dendrogram initialization with default values."""
    assert basic_dendrogram.metric == "euclidean"
    assert basic_dendrogram.method == "average"
    assert basic_dendrogram.orientation == "top"
    assert basic_dendrogram.figsize == (10, 10)
    assert basic_dendrogram.show is False


def test_dendrogram_with_dtm(sample_dtm):
    """Test Dendrogram creation with DTM input."""
    dendrogram = Dendrogram(dtm=sample_dtm)
    dendrogram()
    assert dendrogram.dtm is not None
    assert dendrogram.labels == sample_dtm.labels


def test_dendrogram_with_dataframe(sample_dtm):
    """Test Dendrogram creation with DataFrame input."""
    df = sample_dtm.to_df()
    labels = df.columns.tolist()
    df.index.name = "terms"
    df = df.T
    dendrogram = Dendrogram(dtm=df, labels=labels)
    assert dendrogram.dtm is not None
    assert len(dendrogram.labels) == 5


def test_dendrogram_with_array(sample_dtm):
    """Test Dendrogram creation with numpy array input."""
    df = sample_dtm.to_df()
    labels = df.columns.tolist()
    df.index.name = "terms"
    data = df.T.to_numpy()
    dendrogram = Dendrogram(dtm=data, labels=labels)
    assert dendrogram.dtm is not None
    assert len(dendrogram.labels) == 5


def test_dendrogram_custom_labels():
    """Test Dendrogram with custom labels."""
    data = np.array([[1, 2], [3, 4]])
    custom_labels = ["A", "B"]
    dendrogram = Dendrogram(dtm=data, labels=custom_labels)
    assert dendrogram.labels == custom_labels


def test_dendrogram_no_dtm():
    """Test error handling when no DTM is provided."""
    dendrogram = Dendrogram()
    with pytest.raises(LexosException) as exc_info:
        dendrogram()
    assert "You must provide a document-term matrix" in str(exc_info.value)


@pytest.mark.parametrize(
    "metric,method",
    [
        ("euclidean", "single"),
        ("cosine", "complete"),
        ("correlation", "average"),
    ],
)
def test_dendrogram_different_metrics(sample_dtm, metric, method):
    """Test Dendrogram with different metrics and methods.

    Args:
        metric: Distance metric to use
        method: Linkage method to use
    """
    dendrogram = Dendrogram(dtm=sample_dtm, metric=metric, method=method)
    dendrogram()
    assert isinstance(dendrogram.fig, plt.Figure)


def test_dendrogram_show_option(sample_dtm):
    """Test Dendrogram show option behavior."""
    # Test with show=False
    dendrogram_hidden = Dendrogram(dtm=sample_dtm, show=False)
    dendrogram_hidden()
    assert dendrogram_hidden.fig is not None

    # Test with show=True
    dendrogram_shown = Dendrogram(dtm=sample_dtm, show=True)
    dendrogram_shown()
    assert dendrogram_shown.fig is not None


def test_dendrogram_custom_figure_options(sample_dtm):
    """Test Dendrogram with custom figure options."""
    dendrogram = Dendrogram(
        dtm=sample_dtm,
        figsize=(15, 15),
        title="Test Dendrogram",
        leaf_rotation=45,
        leaf_font_size=12,
    )
    dendrogram()
    assert dendrogram.fig is not None
    assert dendrogram.figsize == (15, 15)


def test_call_with_no_dtm():
    """Test calling dendrogram with no DTM raises exception."""
    dendrogram = Dendrogram()
    with pytest.raises(LexosException) as exc_info:
        dendrogram()
    assert "You must provide a document-term matrix" in str(exc_info.value)


def test_call_with_dtm(sample_dtm):
    """Test calling dendrogram with DTM input."""
    dendrogram = Dendrogram(dtm=sample_dtm)
    dendrogram()
    assert isinstance(dendrogram.fig, plt.Figure)
    assert dendrogram.labels == ["Doc1", "doc2", "doc3", "doc4", "doc5"]


def test_call_with_dataframe():
    """Test calling dendrogram with pandas DataFrame input."""
    df = pd.DataFrame({"doc1": [1, 2, 3], "doc2": [4, 5, 6], "doc3": [7, 8, 9]})
    dendrogram = Dendrogram(dtm=df)
    dendrogram()
    assert isinstance(dendrogram.fig, plt.Figure)
    assert dendrogram.labels == ["doc1", "doc2", "doc3"]


def test_call_with_numpy_array():
    """Test calling dendrogram with numpy array input."""
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dendrogram = Dendrogram(dtm=array)
    dendrogram()
    assert isinstance(dendrogram.fig, plt.Figure)
    assert dendrogram.labels == ["Doc1", "Doc2", "Doc3"]


def test_call_with_custom_labels():
    """Test calling dendrogram with custom labels."""
    array = np.array([[1, 2], [3, 4]])
    custom_labels = ["A", "B"]
    dendrogram = Dendrogram(dtm=array, labels=custom_labels)
    dendrogram()
    assert dendrogram.labels == custom_labels


def test_call_labels_mismatch_after_matrix_creation():
    """Test LexosException on line 159 when labels don't match matrix shape."""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    dendrogram = Dendrogram(dtm=data, labels=["Doc1", "Doc2", "Doc3"])

    # For direct coverage, we can simulate the mismatch.
    dendrogram.labels = [
        "Doc1",
        "Doc2",
    ]  # Now labels length is 2, but matrix shape is 3

    with pytest.raises(LexosException) as exc_info:
        dendrogram()
    assert "The number of labels must match the number of documents." in str(
        exc_info.value
    )


def test_get_valid_matrix_dataframe_non_numeric():
    """Test that a pandas DataFrame with non-numeric values raises LexosException (line 215)."""
    # Create a DataFrame with a non-numeric column
    df_non_numeric = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    dendrogram = Dendrogram(dtm=df_non_numeric)
    with pytest.raises(
        LexosException,
        match="The document-term matrix must contain only numeric values.",
    ):
        dendrogram._get_valid_matrix()


def test_get_valid_matrix_list_single_doc():
    """Test that a list of lists with a single document raises LexosException (line 225)."""
    single_doc_list = [[1, 2, 3]]
    dendrogram = Dendrogram(dtm=single_doc_list)
    with pytest.raises(
        LexosException,
        match="The document-term matrix must have more than one document.",
    ):
        dendrogram._get_valid_matrix()


def test_get_valid_matrix_unsupported_type():
    """Test that an unsupported DTM type raises LexosException (line 238)."""
    unsupported_dtm = "this is not a matrix"
    dendrogram = Dendrogram(dtm=unsupported_dtm)
    with pytest.raises(LexosException, match="Unsupported document-term matrix type."):
        dendrogram._get_valid_matrix()


def test_dendrogram_list_with_non_numeric_values():
    """Test that a list of lists with non-numeric values raises an error."""
    bad_matrix = [[1, 2], ["a", 4]]
    dendrogram = Dendrogram(dtm=bad_matrix)
    with pytest.raises(LexosException, match="must contain only numeric values"):
        dendrogram._get_valid_matrix()


def test_dendrogram_numpy_array_with_non_numeric_values():
    """Test that a numpy array with non-numeric values raises a LexosException."""
    # Forcefully set dtype to object to simulate non-numeric data
    bad_array = np.array([["a", 1], ["b", 2]], dtype=object)
    dendrogram = Dendrogram(dtm=bad_array)
    with pytest.raises(LexosException, match="must contain only numeric values"):
        dendrogram._get_valid_matrix()


def test_dendrogram_string_matrix():
    """Test Dendrogram with an invalid matrix of strings."""
    data = np.array([["invalid", "invalid"], ["invalid", "invalid"]])
    custom_labels = ["A", "B"]
    with pytest.raises(LexosException):
        dendrogram = Dendrogram(dtm=data, labels=custom_labels)
        _ = dendrogram._get_valid_matrix()


def test_dendrogram_char_matrix():
    """Test Dendrogram with an invalid matrix of characters."""
    data = np.array([["a", "b"], ["c", "d"]])
    custom_labels = ["A", "B"]
    with pytest.raises(LexosException):
        dendrogram = Dendrogram(dtm=data, labels=custom_labels)
        _ = dendrogram._get_valid_matrix()


def test_dendrogram_1d_array():
    """Test Dendrogram with an invalid 1D array."""
    data = np.array([[1, 2, 3]])
    custom_labels = ["A"]
    with pytest.raises(LexosException):
        dendrogram = Dendrogram(dtm=data, labels=custom_labels)
        _ = dendrogram._get_valid_matrix()


def test_dendrogram_unequal_labels():
    """Test Dendrogram with mis-matched matrix and label sizes."""
    data = np.array([[1, 2], [3, 4]])
    custom_labels = ["A"]
    with pytest.raises(LexosException):
        dendrogram = Dendrogram(dtm=data, labels=custom_labels)
        _ = dendrogram._get_valid_matrix()


def test_call_with_kwargs():
    """Test calling dendrogram with keyword arguments."""
    array = np.array([[1, 2], [3, 4]])
    dendrogram = Dendrogram(dtm=array)
    dendrogram(title="Test Title", figsize=(15, 15))
    assert dendrogram.title == "Test Title"
    assert dendrogram.figsize == (15, 15)


def test_call_show_behavior():
    """Test show parameter behavior."""
    array = np.array([[1, 2], [3, 4]])

    # Test with show=False
    dendrogram_hidden = Dendrogram(dtm=array, show=False)
    dendrogram_hidden()
    assert dendrogram_hidden.fig is not None

    # Test with show=True
    dendrogram_shown = Dendrogram(dtm=array, show=True)
    dendrogram_shown()
    assert dendrogram_shown.fig is not None


@pytest.mark.parametrize(
    "metric,method",
    [("euclidean", "single"), ("cosine", "complete"), ("correlation", "average")],
)
def test_call_different_metrics(sample_dtm, metric, method):
    """Test dendrogram with different metrics and methods.

    Args:
        metric: Distance metric to use
        method: Linkage method to use
        sample_dtm: Sample DTM fixture
    """
    dendrogram = Dendrogram(dtm=sample_dtm, metric=metric, method=method)
    dendrogram()
    assert isinstance(dendrogram.fig, plt.Figure)


def test_call_with_custom_figure_options(sample_dtm):
    """Test dendrogram with custom figure options."""
    dendrogram = Dendrogram(
        dtm=sample_dtm,
        figsize=(15, 15),
        title="Test Dendrogram",
        leaf_rotation=45,
        leaf_font_size=12,
        orientation="left",
    )
    dendrogram()
    assert isinstance(dendrogram.fig, plt.Figure)
    assert dendrogram.figsize == (15, 15)
    assert dendrogram.title == "Test Dendrogram"


def test_get_valid_matrix_basic(sample_dtm):
    """Test basic matrix conversion from DTM."""
    dendrogram = Dendrogram(dtm=sample_dtm)
    result = dendrogram._get_valid_matrix()

    assert isinstance(result, pd.DataFrame)
    assert result.index.tolist() == ["Doc1", "doc2", "doc3", "doc4", "doc5"]
    assert result.columns.tolist() == [
        "10term",
        "2term",
        "Term3",
        "term1",
        "alert",
        "kitten",
    ]


def test_get_valid_matrix_from_dtm_shape(sample_dtm):
    """Test matrix shape after conversion."""
    dendrogram = Dendrogram(dtm=sample_dtm)
    result = dendrogram._get_valid_matrix()

    # Should be transposed from original 6x5 to 5x6
    assert result.shape == (5, 6)


def test_get_valid_matrix_values(sample_dtm):
    """Test matrix values after conversion."""
    dendrogram = Dendrogram(dtm=sample_dtm)
    result = dendrogram._get_valid_matrix()

    # Check first row values
    assert list(result.iloc[0]) == [0, 0, 0, 0, 1, 1]
    # Check second row values
    assert list(result.iloc[1]) == [0, 0, 0, 1, 0, 0]


def test_get_valid_matrix_from_dtm_single_doc():
    """Test matrix conversion with single document."""
    single_doc_dtm = DTM()
    single_doc_dtm(
        docs=[nlp("kitten alert")],
        labels=["Doc1"],
    )
    # TODO: Test with a dataframe and a numpy array
    with pytest.raises(LexosException):
        dendrogram = Dendrogram(dtm=single_doc_dtm)
        _ = dendrogram._get_valid_matrix()


def test_get_valid_matrix_from_df_single_doc():
    """Test matrix conversion with single document."""
    single_doc_dtm = DTM()
    single_doc_dtm(
        docs=[nlp("kitten alert")],
        labels=["Doc1"],
    )
    df = single_doc_dtm.to_df()
    labels = df.columns.tolist()
    df.index.name = "terms"
    dtm = df.T

    with pytest.raises(LexosException):
        dendrogram = Dendrogram(dtm=dtm, labels=labels)
        _ = dendrogram._get_valid_matrix()


def test_get_valid_matrix_from_array_single_doc():
    """Test matrix conversion with single document."""
    single_doc_dtm = DTM()
    single_doc_dtm(
        docs=[nlp("kitten alert")],
        labels=["Doc1"],
    )
    df = single_doc_dtm.to_df()
    labels = df.columns.tolist()
    df.index.name = "terms"
    dtm = df.T.to_numpy()

    with pytest.raises(LexosException):
        dendrogram = Dendrogram(dtm=dtm, labels=labels)
        _ = dendrogram._get_valid_matrix()


def test_save_basic(sample_dendrogram, tmp_path):
    """Test basic figure saving functionality.

    Args:
        sample_dendrogram: Fixture with prepared dendrogram
        tmp_path: pytest fixture providing temporary directory
    """
    save_path = tmp_path / "test_dendrogram.png"
    sample_dendrogram.save(save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_save_with_string_path(sample_dendrogram, tmp_path):
    """Test saving with string path instead of Path object.

    Args:
        sample_dendrogram: Fixture with prepared dendrogram
        tmp_path: pytest fixture providing temporary directory
    """
    save_path = str(tmp_path / "test_dendrogram.png")
    sample_dendrogram.save(save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0


def test_save_different_formats(sample_dendrogram, tmp_path):
    """Test saving in different file formats.

    Args:
        sample_dendrogram: Fixture with prepared dendrogram
        tmp_path: pytest fixture providing temporary directory
    """
    formats = [".png", ".pdf", ".svg", ".jpg"]
    for fmt in formats:
        save_path = tmp_path / f"test_dendrogram{fmt}"
        sample_dendrogram.save(save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


def test_save_no_figure():
    """Test save behavior when no figure exists."""
    dendrogram = Dendrogram()  # Create empty dendrogram
    with pytest.raises(AttributeError):
        dendrogram.save("test.png")


def test_save_invalid_path(sample_dendrogram):
    """Test save behavior with invalid path.

    Args:
        sample_dendrogram: Fixture with prepared dendrogram
    """
    with pytest.raises(LexosException):
        sample_dendrogram.save("")  # Empty path


def test_save_with_spaces(sample_dendrogram, tmp_path):
    """Test saving to path with spaces.

    Args:
        sample_dendrogram: Fixture with prepared dendrogram
        tmp_path: pytest fixture providing temporary directory
    """
    save_path = tmp_path / "test dendrogram with spaces.png"
    sample_dendrogram.save(save_path)
    assert save_path.exists()


def test_showfig_basic(sample_dendrogram):
    """Test basic figure display functionality.

    Args:
        sample_dendrogram: Fixture with prepared dendrogram
    """
    result = sample_dendrogram.showfig()
    assert isinstance(result, plt.Figure)
    assert result == sample_dendrogram.fig


def test_showfig_no_figure():
    """Test showfig behavior when no figure exists."""
    dendrogram = Dendrogram()  # Create empty dendrogram without figure
    assert dendrogram.showfig() is None


def test_showfig_after_close(sample_dendrogram):
    """Test showfig after closing the figure.

    Args:
        sample_dendrogram: Fixture with prepared dendrogram
    """
    plt.close(sample_dendrogram.fig)
    result = sample_dendrogram.showfig()
    assert isinstance(result, plt.Figure)
    assert result == sample_dendrogram.fig


def test_showfig_multiple_calls(sample_dendrogram):
    """Test multiple calls to showfig return same figure.

    Args:
        sample_dendrogram: Fixture with prepared dendrogram
    """
    first_call = sample_dendrogram.showfig()
    second_call = sample_dendrogram.showfig()
    assert first_call is second_call
    assert isinstance(first_call, plt.Figure)
