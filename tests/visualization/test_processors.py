"""test_processors.py.

Last Update: March 1, 2025
"""

import numpy as np
import pandas as pd
import pytest
import spacy
from scipy.sparse import csr_matrix

from lexos.dtm import DTM
from lexos.visualization.processors import (
    filter_docs,
    get_rows,
    multicloud_processor,
    process_dataframe,
    process_docs,
    process_dtm,
    process_item,
    process_list,
)

# Fixtures


@pytest.fixture
def nlp():
    """Create spacy nlp object for testing.

    Returns:
        Language: spaCy Language object
    """
    return spacy.blank("en")


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing.

    Returns:
        pd.DataFrame: Sample DataFrame with test data
    """
    data = {"doc1": [1, 2, 3], "doc2": [4, 5, 6], "doc3": [7, 8, 9]}
    return pd.DataFrame(data, index=["term1", "term2", "term3"])


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
def sample_dtm2():
    """Create sample DTM for testing.

    Returns:
        DTM: Sample DTM with test data
    """
    dtm = DTM()

    # Create sample data
    data = np.array(
        [[15, 10, 13, 8], [10, 12, 11, 3], [12, 11, 10, 9], [5, 14, 23, 12]]
    )
    dtm.doc_term_matrix = csr_matrix(data)

    # Mock vectorizer
    class MockVectorizer:
        terms_list = ["term1", "term2", "term3", "term4"]

    dtm.vectorizer = MockVectorizer()

    # Set labels
    dtm.labels = ["doc1", "doc2", "doc3", "doc4"]

    return dtm


@pytest.fixture
def sample_docs(nlp):
    """Create sample spaCy docs for testing.

    Args:
        nlp: spaCy Language object

    Returns:
        list: List of Doc objects
    """
    texts = ["hello world", "test document", "sample text"]
    return [nlp(text) for text in texts]


# Tests


def test_filter_docs_no_filter(sample_df):
    """Test filter_docs with no filtering applied."""
    result = filter_docs(sample_df)
    pd.testing.assert_frame_equal(result, sample_df)


def test_filter_docs_with_string_labels(sample_df):
    """Test filter_docs with string document labels."""
    docs = ["doc1", "doc3"]
    result = filter_docs(sample_df, docs)
    expected = sample_df[["doc1", "doc3"]]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_docs_with_indices(sample_df):
    """Test filter_docs with integer indices."""
    docs = [0, 2]
    result = filter_docs(sample_df, docs)
    expected = sample_df.iloc[:, [0, 2]]
    pd.testing.assert_frame_equal(result, expected)


def test_filter_docs_empty_list(sample_df):
    """Test filter_docs with empty docs list."""
    docs = []
    result = filter_docs(sample_df, docs)
    pd.testing.assert_frame_equal(result, sample_df)


def test_filter_docs_none(sample_df):
    """Test filter_docs with None as docs parameter."""
    result = filter_docs(sample_df, None)
    pd.testing.assert_frame_equal(result, sample_df)


def test_filter_docs_single_doc():
    """Test filter_docs with single document."""
    df = pd.DataFrame({"doc1": [1, 2, 3]}, index=["term1", "term2", "term3"])
    docs = ["doc1"]
    result = filter_docs(df, docs)
    pd.testing.assert_frame_equal(result, df)


@pytest.mark.parametrize(
    "docs,expected_cols", [(["doc1", "doc2"], 2), ([0, 1, 2], 3), (None, 3)]
)
def test_filter_docs_output_shape(sample_df, docs, expected_cols):
    """Test output shape of filtered DataFrame.

    Args:
        sample_df: Sample DataFrame fixture
        docs: List of documents to filter
        expected_cols: Expected number of columns in result
    """
    result = filter_docs(sample_df, docs)
    assert result.shape[1] == expected_cols


def test_process_dataframe_basic(sample_df):
    """Test basic processing of DataFrame without document filtering."""
    result = process_dataframe(sample_df)
    expected = {"term1": 12, "term2": 15, "term3": 18}
    assert result == expected


def test_process_dataframe_with_string_filter(sample_df):
    """Test DataFrame processing with string-based document filter."""
    result = process_dataframe(sample_df, docs=["doc1", "doc2"])
    expected = {"term1": 5, "term2": 7, "term3": 9}
    assert result == expected


def test_process_dataframe_with_index_filter(sample_df):
    """Test DataFrame processing with index-based document filter."""
    result = process_dataframe(sample_df, docs=[0, 2])
    expected = {"term1": 8, "term2": 10, "term3": 12}
    assert result == expected


def test_process_dataframe_single_doc(sample_df):
    """Test processing with single document selection."""
    result = process_dataframe(sample_df, docs="doc1")
    expected = {"term1": 1, "term2": 2, "term3": 3}
    assert result == expected


def test_process_dataframe_zero_counts():
    """Test handling of zero counts in DataFrame."""
    df = pd.DataFrame(
        {"doc1": [0, 0, 0], "doc2": [1, 0, 2]}, index=["term1", "term2", "term3"]
    )
    result = process_dataframe(df)
    expected = {"term1": 1, "term3": 2}
    assert result == expected
    assert "term2" not in result


@pytest.mark.parametrize(
    "docs,expected_terms",
    [
        (["doc1"], {"term1": 1, "term2": 2, "term3": 3}),
        (["doc1", "doc2"], {"term1": 5, "term2": 7, "term3": 9}),
        (["doc3"], {"term1": 7, "term2": 8, "term3": 9}),
        (None, {"term1": 12, "term2": 15, "term3": 18}),
    ],
)
def test_process_dataframe_term_counts(sample_df, docs, expected_terms):
    """Test number of terms in result with different document selections.

    Args:
        sample_dtm: Sample DTM fixture
        docs: List of documents to filter
        expected_terms: Expected number of terms in result
    """
    result = process_dataframe(sample_df, docs=docs)
    assert result == expected_terms


def test_process_dtm_basic(sample_dtm):
    """Test basic DTM processing without document filtering."""
    result = process_dtm(sample_dtm)
    expected = {
        "term1": 3,  # Sum of first row
        "term2": 3,  # Sum of second row
        "term3": 4,  # Sum of third row
    }
    assert result == expected


def test_process_dtm_with_single_doc(sample_dtm):
    """Test DTM processing with single document selection."""
    result = process_dtm(sample_dtm, docs="doc1")
    expected = {
        "term1": 1,
        "term3": 3,
    }
    assert result == expected


def test_process_dtm_with_multiple_docs(sample_dtm):
    """Test DTM processing with multiple document selection."""
    result = process_dtm(sample_dtm, docs=["doc1", "doc2"])
    expected = {"term1": 1, "term2": 2, "term3": 4}
    assert result == expected


def test_process_dtm_with_index_filter(sample_dtm):
    """Test DTM processing with index-based document filter."""
    result = process_dtm(sample_dtm, docs=[0, 2])
    expected = {"term1": 3, "term2": 1, "term3": 3}
    assert result == expected


@pytest.mark.parametrize(
    "docs,expected_terms",
    [
        ("doc1", {"term1": 1, "term3": 3}),  # Single doc
        (["doc1", "doc2"], {"term1": 1, "term2": 2, "term3": 4}),  # Multiple docs
        (None, {"term1": 3, "term2": 3, "term3": 4}),  # All docs
        ([0], {"term1": 1, "term3": 3}),  # Single doc by index
    ],
)
def test_process_dtm_term_counts(sample_dtm, docs, expected_terms):
    """Test number of terms in result with different document selections.

    Args:
        sample_dtm: Sample DTM fixture
        docs: Documents to filter
        expected_terms: Expected number of non-zero terms
    """
    result = process_dtm(sample_dtm, docs=docs)
    # Only count terms with non-zero frequencies
    # actual_terms = len([v for v in result.values() if v > 0])
    assert result == expected_terms


def test_process_dtm_zero_counts():
    """Test handling of zero counts in DTM."""
    dtm = DTM()
    data = np.array([[0, 0, 0], [1, 0, 2], [0, 0, 0]])
    dtm.doc_term_matrix = csr_matrix(data)

    # Mock vectorizer
    class MockVectorizer:
        terms_list = ["term1", "term2", "term3"]

    dtm.vectorizer = MockVectorizer()

    # Set labels
    dtm.labels = ["doc1", "doc2", "doc3"]

    result = process_dtm(dtm)

    expected = {"term1": 1, "term3": 2}
    assert result == expected


def test_process_list_strings():
    """Test processing list of string lists."""
    data = [["hello", "world"], ["test", "doc"]]
    result = process_list(data, docs=None)

    expected = {"hello": 1, "world": 1, "test": 1, "doc": 1}
    assert result == expected


def test_process_list_tokens(nlp):
    """Test processing list of token lists."""
    doc1 = nlp("hello world")
    doc2 = nlp("test doc")
    data = [[token for token in doc1], [token for token in doc2]]

    result = process_list(data, docs=None)
    print(result)
    # Convert tokens to strings
    result = {k.text: v for k, v in result.items()}
    expected = {"hello": 1, "world": 1, "test": 1, "doc": 1}
    assert result == expected


def test_process_list_docs(sample_docs):
    """Test processing list of Doc objects."""
    result = process_list(sample_docs, docs=None)
    # Convert tokens to strings
    result = {k.text: v for k, v in result.items()}
    expected = {
        "hello": 1,
        "world": 1,
        "test": 1,
        "document": 1,
        "sample": 1,
        "text": 1,
    }
    assert result == expected


def test_process_list_with_doc_filter(sample_docs):
    """Test processing with document filtering."""
    result = process_list(sample_docs, docs=[0, 2])
    expected = {"hello": 1, "world": 1, "sample": 1, "text": 1}
    assert result == expected
    assert "test" not in result
    assert "document" not in result


def test_process_list_spans(nlp):
    """Test processing list of Span objects."""
    doc = nlp("hello world test document")
    spans = [doc[0:2], doc[2:4]]  # Create spans from doc
    result = process_list(spans, docs=None)
    # Convert tokens to strings
    result = {k.text: v for k, v in result.items()}
    expected = {"hello": 1, "world": 1, "test": 1, "document": 1}
    assert result == expected


@pytest.mark.parametrize(
    "docs,expected_count",
    [
        ([0], 2),  # First document only
        ([1], 2),  # Second document only
        ([0, 1], 4),  # Both documents
        (None, 4),  # All documents
    ],
)
def test_process_list_doc_filtering(sample_docs, docs, expected_count):
    """Test document filtering with different selections.

    Args:
        sample_docs: Sample docs fixture
        docs: Document indices to filter
        expected_count: Expected number of terms in result
    """
    result = process_list(sample_docs[:2], docs=docs)
    assert len(result) == expected_count


def test_process_list_empty_docs():
    """Test processing with empty document list."""
    result = process_list([], docs=None)
    assert result == {}


def test_process_list_single_token_docs(nlp):
    """Test processing documents with single tokens."""
    docs = [nlp("hello"), nlp("world")]
    result = process_list(docs, docs=None)
    # Convert tokens to strings
    result = {k.text: v for k, v in result.items()}

    expected = {"hello": 1, "world": 1}
    assert result == expected


@pytest.fixture
def sample_spans(nlp):
    """Create sample spaCy spans for testing.

    Args:
        nlp: spaCy Language object

    Returns:
        list: List of Span objects
    """
    doc = nlp("hello world test document sample text")
    return [doc[0:2], doc[2:4], doc[4:6]]


def test_process_docs_basic(sample_docs):
    """Test basic processing of Doc objects."""
    result = process_docs(sample_docs, docs=None)

    expected = {
        "hello": 1,
        "world": 1,
        "test": 1,
        "document": 1,
        "sample": 1,
        "text": 1,
    }
    assert result == expected


def test_process_docs_with_spans(sample_spans):
    """Test processing of Span objects."""
    result = process_docs(sample_spans, docs=None)

    expected = {
        "hello": 1,
        "world": 1,
        "test": 1,
        "document": 1,
        "sample": 1,
        "text": 1,
    }
    assert result == expected


def test_process_docs_with_filtering(sample_docs):
    """Test document filtering."""
    result = process_docs(sample_docs, docs=[0, 2])

    expected = {"hello": 1, "world": 1, "sample": 1, "text": 1}
    assert result == expected
    assert "test" not in result
    assert "document" not in result


@pytest.mark.parametrize(
    "docs,expected_terms",
    [
        ([0], 2),  # First document only
        ([1], 2),  # Second document only
        ([0, 1], 4),  # First two documents
        (None, 6),  # All documents
    ],
)
def test_process_docs_term_counts(sample_docs, docs, expected_terms):
    """Test term counting with different document selections.

    Args:
        sample_docs: Sample docs fixture
        docs: Document indices to filter
        expected_terms: Expected number of terms in result
    """
    result = process_docs(sample_docs, docs=docs)
    assert len(result) == expected_terms


def test_process_docs_empty_list():
    """Test processing empty document list."""
    result = process_docs([], docs=None)
    assert result == {}


def test_process_docs_single_token(nlp):
    """Test processing documents with single tokens."""
    docs = [nlp("hello"), nlp("world")]
    result = process_docs(docs, docs=None)

    expected = {"hello": 1, "world": 1}
    assert result == expected


def test_process_docs_repeated_terms(nlp):
    """Test handling of repeated terms."""
    docs = [nlp("hello hello"), nlp("world world")]
    result = process_docs(docs, docs=None)

    expected = {"hello": 2, "world": 2}
    assert result == expected


def test_process_item_string_list():
    """Test processing list of strings."""
    data = ["hello", "world", "hello"]
    result = process_item(data)

    expected = {"hello": 2, "world": 1}
    assert result == expected


def test_process_item_token_list(nlp):
    """Test processing list of tokens."""
    doc = nlp("hello world hello")
    tokens = [token for token in doc]
    result = process_item(tokens)

    expected = {"hello": 2, "world": 1}
    assert result == expected


def test_process_item_doc(nlp):
    """Test processing single Doc object."""
    doc = nlp("hello world hello")
    result = process_item(doc)

    expected = {"hello": 2, "world": 1}
    assert result == expected


def test_process_item_span(nlp):
    """Test processing single Span object."""
    doc = nlp("hello world test")
    span = doc[0:2]  # Only "hello world"
    result = process_item(span)

    expected = {"hello": 1, "world": 1}
    assert result == expected


def test_process_item_empty_list():
    """Test processing empty list."""
    with pytest.raises(IndexError):
        process_item([])


def test_process_item_single_token(nlp):
    """Test processing list with single token."""
    doc = nlp("hello")
    tokens = [token for token in doc]
    result = process_item(tokens)

    expected = {"hello": 1}
    assert result == expected


@pytest.mark.parametrize(
    "text,expected_count",
    [
        ("word word word", {"word": 3}),
        ("a b c", {"a": 1, "b": 1, "c": 1}),
        ("test", {"test": 1}),
    ],
)
def test_process_item_various_texts(nlp, text, expected_count):
    """Test processing various text patterns.

    Args:
        nlp: spaCy Language object
        text: Input text to process
        expected_count: Expected term frequency dictionary
    """
    doc = nlp(text)
    result = process_item(doc)
    assert result == expected_count


def test_process_item_mixed_case(nlp):
    """Test processing text with mixed case."""
    doc = nlp("Hello WORLD hello World")
    result = process_item(doc)

    # Case is preserved
    expected = {"Hello": 1, "WORLD": 1, "hello": 1, "World": 1}
    assert result == expected


def test_process_item_punctuation(nlp):
    """Test processing text with punctuation."""
    doc = nlp("hello, world! hello.")
    result = process_item(doc)

    expected = {"hello": 2, ",": 1, "world": 1, "!": 1, ".": 1}
    assert result == expected


def test_multicloud_processor_dtm(sample_dtm2):
    """Test processing DTM input."""
    result = multicloud_processor(sample_dtm2)

    assert isinstance(result, list)
    assert all(isinstance(d, dict) for d in result)
    assert len(result) > 0
    # Check non-zero values only
    assert all(v > 0 for d in result for v in d.values())


def test_multicloud_processor_dataframe(sample_dtm2):
    """Test processing DataFrame input."""
    df = sample_dtm2.to_df()
    result = multicloud_processor(df)

    assert isinstance(result, list)
    assert all(isinstance(d, dict) for d in result)
    assert all(v > 0 for d in result for v in d.values())


def test_multicloud_processor_docs(sample_docs):
    """Test processing Doc objects."""
    result = multicloud_processor(sample_docs)

    assert isinstance(result, list)
    assert len(result) == len(sample_docs)
    assert all(isinstance(d, dict) for d in result)


def test_multicloud_processor_with_filtering(sample_dtm2):
    """Test document filtering."""
    result = multicloud_processor(sample_dtm2, docs=[0, 2])

    assert isinstance(result, list)
    assert all(isinstance(d, dict) for d in result)
    assert len(result) == 2


def test_multicloud_processor_string_lists():
    """Test processing lists of string lists."""
    data = [["hello", "world"], ["test", "doc"]]
    result = multicloud_processor(data)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(d, dict) for d in result)


def test_multicloud_processor_token_lists(nlp):
    """Test processing lists of token lists."""
    docs = [nlp("hello world"), nlp("test doc")]
    doc1 = [token for token in docs[0][0:1]]
    doc2 = [token for token in docs[1][1:2]]
    data = [doc1, doc2]
    result = multicloud_processor(data)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(d, dict) for d in result)


def test_multicloud_processor_dict_list():
    """Test processing list of dictionaries."""
    data = [{"word1": 1, "word2": 2}, {"word3": 3, "word4": 4}]
    result = multicloud_processor(data)

    assert result == data


@pytest.mark.parametrize("docs,expected_len", [([0], 1), ([0, 1], 2), (None, 3)])
def test_multicloud_processor_doc_filtering(sample_docs, docs, expected_len):
    """Test document filtering with different selections.

    Args:
        sample_docs: Sample docs fixture
        docs: Document indices to filter
        expected_len: Expected number of documents in result
    """
    result = multicloud_processor(sample_docs, docs=docs)
    assert len(result) == expected_len


def test_get_rows_even_division():
    """Test get_rows with list length divisible by row size.

    Tests that documents are correctly divided into rows when the list
    length is evenly divisible by the number of columns.
    """
    data = [1, 2, 3, 4, 5, 6]
    rows = list(get_rows(data, 2))

    assert len(rows) == 3
    assert rows == [[1, 2], [3, 4], [5, 6]]


def test_get_rows_uneven_division():
    """Test get_rows with list length not divisible by row size.

    Tests that documents are correctly divided when the list length
    is not evenly divisible by the number of columns.
    """
    data = [1, 2, 3, 4, 5]
    rows = list(get_rows(data, 2))

    assert len(rows) == 3
    assert rows == [[1, 2], [3, 4], [5]]


def test_get_rows_single_column():
    """Test get_rows with single column configuration.

    Tests that documents are correctly divided when each row
    contains only one item.
    """
    data = [1, 2, 3]
    rows = list(get_rows(data, 1))

    assert len(rows) == 3
    assert rows == [[1], [2], [3]]


def test_get_rows_empty_list():
    """Test get_rows with empty list.

    Tests that the generator yields no rows when given an empty list.
    """
    data = []
    rows = list(get_rows(data, 2))

    assert len(rows) == 0
    assert rows == []


@pytest.mark.parametrize(
    "data,n,expected",
    [
        ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
        ([1, 2, 3], 2, [[1, 2], [3]]),
        ([1, 2, 3, 4, 5], 3, [[1, 2, 3], [4, 5]]),
        ([1], 1, [[1]]),
    ],
)
def test_get_rows_different_sizes(data, n, expected):
    """Test get_rows with different list and row sizes.

    Args:
        data: Input list to divide into rows
        n: Number of columns per row
        expected: Expected row configuration
    """
    rows = list(get_rows(data, n))
    assert rows == expected


def test_get_rows_iterator_type():
    """Test that get_rows returns an iterator.

    Verifies that the function returns an iterator object that can be
    consumed multiple times.
    """
    data = [1, 2, 3, 4]
    rows = get_rows(data, 2)

    assert hasattr(rows, "__iter__")
    assert hasattr(rows, "__next__")
