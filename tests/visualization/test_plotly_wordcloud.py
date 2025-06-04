"""test_plotly_wordcloud.py.

Last Update: March 4, 2025

Note: Only tests a few input types since the tests for all input types are implemented in test_cloud.py. Since all cloud input methods now
return a list of term-count dicts, it might be a good idea to remove
all input tests from test files for individual cloud types and instead
test the conversion to term-count dicts only in test_processors.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import spacy
from scipy.sparse import csr_matrix
from wordcloud import WordCloud

from lexos.dtm import DTM
from lexos.visualization.plotly_wordcloud import plotly_wordcloud
from lexos.visualization.cloud import wordcloud 

# Fixtures

@pytest.fixture
def nlp():
    """Create spacy nlp object for testing.

    Returns:
        Language: spaCy Language object
    """
    return spacy.blank("en")

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing.

    Returns:
        pd.DataFrame: Sample DataFrame with term frequencies
    """
    return pd.DataFrame({
        'doc1': [1, 2],
        'doc2': [3, 4]
    }, index=['term1', 'term2'])

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


# Tests

def test_wordcloud_string_input(sample_text):
    """Test wordcloud generation from string input."""
    wc = plotly_wordcloud(sample_text, show=False)
    assert isinstance(wc, go.Figure)

def test_wordcloud_doc_input(nlp, sample_text):
    """Test wordcloud generation from spaCy Doc."""
    doc = nlp(sample_text)
    wc = plotly_wordcloud(doc, show=False)
    assert isinstance(wc, go.Figure)

def test_wordcloud_dtm_input(sample_dtm):
    """Test wordcloud generation from DTM."""
    wc = plotly_wordcloud(sample_dtm, show=False)
    assert isinstance(wc, go.Figure)

@pytest.fixture
def mock_wordcloud():
    """Create a mock WordCloud instance with predefined layout.

    Returns:
        WordCloud: Mock WordCloud object with test layout data
    """
    wc = WordCloud().generate_from_text("test word cloud")
    # Mock the layout_ property to return test data
    wc.layout_ = [
        (("test", 0.5), 20, (100, 100), None, "#000000"),
        (("word", 0.3), 15, (200, 150), None, "#111111"),
        (("cloud", 0.2), 10, (150, 200), None, "#222222")
    ]
    return wc

def test_plotly_conversion(mock_wordcloud, monkeypatch):
    """Test conversion from WordCloud to Plotly figure."""
    # Mock WordCloud generation to return our mock
    monkeypatch.setattr(WordCloud, "generate_from_text", lambda self, text: mock_wordcloud)

    fig = plotly_wordcloud("test text", show=False)

    # Verify figure type
    assert isinstance(fig, go.Figure)

    # Check scatter trace properties
    scatter = fig.data[0]
    assert isinstance(scatter, go.Scatter)
    assert scatter.mode == "text"

    # Verify word positions
    assert scatter.x == (100, 200, 150)
    assert scatter.y == (100, 150, 200)

    # Verify text properties
    assert scatter.text == ("test", "word", "cloud")
    assert scatter.textfont.size == (20, 15, 10)
    assert scatter.textfont.color == ("#000000", "#111111", "#222222")

def test_layout_options():
    """Test custom layout options."""
    custom_layout = {
        "width": 1000,
        "height": 800,
        "title": "Test Cloud"
    }

    fig = plotly_wordcloud("test", layout=custom_layout, show=False)

    assert fig.layout.width == 1000
    assert fig.layout.height == 800
    assert fig.layout.title.text == "Test Cloud"

def test_frequency_formatting(mock_wordcloud, monkeypatch):
    """Test frequency percentage formatting in hover text."""
    monkeypatch.setattr(WordCloud, "generate_from_text", lambda self, text: mock_wordcloud)

    fig = plotly_wordcloud("test text", show=False)

    hover_text = fig.data[0].hovertext
    assert hover_text == ("test: 50.00%", "word: 30.00%", "cloud: 20.00%")

@pytest.mark.parametrize("show_param", [True, False])
def test_show_parameter(show_param, monkeypatch):
    """Test show parameter behavior.

    Args:
        show_param: Boolean value for show parameter
        monkeypatch: Pytest fixture for mocking
    """
    # Mock plotly show method
    show_called = False
    def mock_show(self):
        nonlocal show_called
        show_called = True

    monkeypatch.setattr(go.Figure, "show", mock_show)

    fig = plotly_wordcloud("test", show=show_param)

    assert isinstance(fig, go.Figure)
    assert show_called == show_param

def test_save_to_file(tmp_path):
    """Test saving figure to HTML file."""
    output_path = tmp_path / "test_cloud.html"

    plotly_wordcloud("test", path=output_path, show=False)

    assert output_path.exists()
    assert output_path.suffix == ".html"

def test_wordcloud_list_of_lists():
    """Test Plotly WordCloud from list of lists."""
    data = [["hello", "world"], ["test", "document"]]
    fig = plotly_wordcloud(data, show=False)
    assert isinstance(fig, go.Figure)

def test_wordcloud_list_of_docs(nlp, sample_text):
    """Test Plotly WordCloud from list of spaCy Docs."""
    doc = nlp(sample_text)
    data = [doc, doc]
    fig = plotly_wordcloud(data, show=False)
    assert isinstance(fig, go.Figure)

def test_wordcloud_list_of_strings():
    """Test Plotly WordCloud from list of strings."""
    data = ["one", "two", "two", "three"]
    fig = plotly_wordcloud(data, show=False)
    assert isinstance(fig, go.Figure)

def test_wordcloud_dict_input():
    """Test Plotly WordCloud from dictionary input."""
    data = {"word": 3, "cloud": 2}
    fig = plotly_wordcloud(data, show=False)
    assert isinstance(fig, go.Figure)

