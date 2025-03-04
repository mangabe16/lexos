"""test_cloud.py.

Last Update: March 1, 2025
"""

import numpy as np
import pytest
import spacy
from pydantic import ValidationError
from scipy.sparse import csr_matrix
from wordcloud import WordCloud

from lexos.dtm import DTM
from lexos.visualization.cloud import multicloud, wordcloud

# Fixtures


@pytest.fixture
def nlp():
    """Create spacy nlp object for testing.

    Returns:
        Language: spaCy Language object
    """
    return spacy.blank("en")


@pytest.fixture
def sample_text():
    """Create sample text for testing.

    Returns:
        str: Sample text
    """
    return "hello world hello test document test"


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
def sample_multicloud_dtm():
    """Create sample DTM for testing multiclouds.

    Returns:
        DTM: Sample DTM with test data
    """
    # data = np.array([[1, 2, 3], [4, 5, 6]])
    # terms = ["term1", "term2"]
    # docs = ["doc1", "doc2", "doc3"]
    # return DTM(matrix=data, terms=terms, labels=docs)
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
def sample_texts():
    """Create sample texts for testing.

    Returns:
        list: List of sample texts
    """
    return ["hello world", "test document", "sample text"]


# Tests


def test_wordcloud_string_input(sample_text):
    """Test wordcloud generation from string input."""
    wc = wordcloud(sample_text, show=False)
    assert isinstance(wc, WordCloud)


def test_wordcloud_doc_input(nlp, sample_text):
    """Test wordcloud generation from spaCy Doc."""
    doc = nlp(sample_text)
    wc = wordcloud(doc, show=False)
    assert isinstance(wc, WordCloud)


def test_wordcloud_with_options():
    """Test wordcloud with custom options."""
    opts = {"background_color": "black", "max_words": 100, "width": 800, "height": 400}
    wc = wordcloud("test text", opts=opts, show=False)
    assert isinstance(wc, WordCloud)
    assert wc.background_color == "black"
    assert wc.max_words == 100


def test_wordcloud_with_round_mask():
    """Test wordcloud with round mask."""
    wc = wordcloud("test text", round=200, show=False)
    assert isinstance(wc, WordCloud)
    assert wc.mask is not None


def test_wordcloud_save_to_file(tmp_path):
    """Test saving wordcloud to file."""
    path = tmp_path / "test_cloud.png"
    wordcloud("test text", path=path, show=True)
    assert path.exists()


def test_wordcloud_dtm_input(sample_dtm):
    """Test wordcloud generation from DTM."""
    wc = wordcloud(sample_dtm, show=False)
    assert isinstance(wc, WordCloud)


@pytest.mark.parametrize(
    "figure_opts", [{"figsize": (10, 5)}, {"dpi": 300}, {"facecolor": "gray"}]
)
def test_wordcloud_figure_options(figure_opts):
    """Test wordcloud with different figure options.

    Args:
        figure_opts: Dictionary of figure options to test
    """
    wc = wordcloud("test text", figure_opts=figure_opts, show=False)
    assert isinstance(wc, WordCloud)


def test_wordcloud_invalid_input():
    """Test wordcloud with invalid input type."""
    with pytest.raises(ValidationError):
        wordcloud(123, show=False)


def test_wordcloud_empty_input():
    """Test wordcloud with empty input."""
    with pytest.raises(ValueError):
        wordcloud("", show=False)


def test_wordcloud_return_object():
    """Test wordcloud return object when show=False."""
    result = wordcloud("test text", show=False)
    assert isinstance(result, WordCloud)
    assert hasattr(result, "to_file")


def test_multicloud_string_input(sample_texts):
    """Test multicloud generation from list of strings."""
    clouds = multicloud(sample_texts, show=False)
    assert isinstance(clouds, list)
    assert all(isinstance(cloud, WordCloud) for cloud in clouds)
    assert len(clouds) == len(sample_texts)


def test_multicloud_dtm_input(sample_multicloud_dtm):
    """Test multicloud generation from DTM."""
    clouds = multicloud(sample_multicloud_dtm, show=False)
    assert isinstance(clouds, list)
    assert all(isinstance(cloud, WordCloud) for cloud in clouds)


def test_multicloud_docs_input(nlp):
    """Test multicloud generation from DTM."""
    docs = [nlp("hello world"), nlp("test document")]
    clouds = multicloud(docs, show=False)
    assert isinstance(clouds, list)
    assert all(isinstance(cloud, WordCloud) for cloud in clouds)


def test_multicloud_list_token_input(nlp):
    """Test multicloud generation from DTM."""
    docs = [nlp("hello world"), nlp("test document")]
    tokens = [[t for t in doc] for doc in docs]
    clouds = multicloud(tokens, show=False)
    assert isinstance(clouds, list)
    assert all(isinstance(cloud, WordCloud) for cloud in clouds)


def test_multicloud_with_options():
    """Test multicloud with custom options."""
    opts = {"background_color": "black", "max_words": 100}
    clouds = multicloud(["test1", "test2"], opts=opts, show=False)
    assert all(cloud.background_color == "black" for cloud in clouds)
    assert all(cloud.max_words == 100 for cloud in clouds)


def test_multicloud_with_round_mask():
    """Test multicloud with round mask."""
    clouds = multicloud(["test1", "test2"], round=200, show=False)
    assert all(cloud.mask is not None for cloud in clouds)


def test_multicloud_save_to_file(tmp_path):
    """Test saving multicloud to file."""
    filename = tmp_path / "test_multicloud.png"
    multicloud(["test1", "test2"], filename=str(filename), show=True)
    assert filename.exists()


def test_multicloud_with_title():
    """Test multicloud with title."""
    clouds = multicloud(["test1", "test2"], title="Test Multiclouds", show=False)
    assert isinstance(clouds, list)


def test_multicloud_with_labels():
    """Test multicloud with custom labels."""
    clouds = multicloud(["test1", "test2"], labels=["Doc1", "Doc2"], show=False)
    assert isinstance(clouds, list)


@pytest.mark.parametrize(
    "ncols,expected_rows",
    [
        (2, 2),  # 3 docs, 2 columns = 2 rows
        (3, 1),  # 3 docs, 3 columns = 1 row
        (1, 3),  # 3 docs, 1 column = 3 rows
    ],
)
def test_multicloud_grid_layout(sample_texts, ncols, expected_rows):
    """Test multicloud grid layout with different column configurations.

    Args:
        sample_texts: List of sample texts
        ncols: Number of columns in grid
        expected_rows: Expected number of rows in grid
    """
    clouds = multicloud(sample_texts, ncols=ncols, show=False)
    assert len(clouds) == len(sample_texts)


def test_multicloud_figure_options():
    """Test multicloud with custom figure options."""
    figure_opts = {"figsize": (10, 5), "dpi": 300}
    clouds = multicloud(["test1", "test2"], figure_opts=figure_opts, show=False)
    assert isinstance(clouds, list)


def test_multicloud_empty_input():
    """Test multicloud with empty input list."""
    with pytest.raises(IndexError):
        multicloud([], show=False)


def test_multicloud_single_doc():
    """Test multicloud with single document."""
    clouds = multicloud(["single text"], show=False)
    assert len(clouds) == 1
    assert isinstance(clouds[0], WordCloud)
