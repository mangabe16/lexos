"""test_dtm.py.

Last Update: Jan 28, 2025
"""

import numpy as np
import pandas as pd
import pytest
import spacy
from natsort import ns
from scipy.sparse import csr_matrix
from textacy.representations.vectorizers import Vectorizer as TextacyVectorizer

from lexos.dtm import DTM
from lexos.exceptions import LexosException

# Fixtures


@pytest.fixture
def nlp():
    return spacy.blank("en")


@pytest.fixture
def sample_docs(nlp):
    texts = ["hello world", "test document"]
    return [nlp(text) for text in texts]


@pytest.fixture
def dtm():
    return DTM(vectorizer=TextacyVectorizer)


@pytest.fixture
def dtm_with_terms(nlp):
    dtm = DTM()
    dtm(
        docs=[nlp("term2"), nlp("term1"), nlp("Term3"), nlp("10term"), nlp("2term")],
        labels=["doc1", "doc2", "doc3", "doc4", "doc5"],
    )
    return dtm


@pytest.fixture
def mock_dtm():
    """Create DTM with mocked vocabulary terms."""
    dtm = DTM()

    class MockVectorizer:
        vocabulary_terms = {"term2": 5, "term1": 3, "Term3": 7, "10term": 2, "2term": 4}

    dtm.vectorizer = MockVectorizer()
    return dtm


@pytest.fixture
def mock_df_dtm():
    """Create DTM with sample data."""
    dtm = DTM()

    # Create sample data
    data = np.array([[1, 2], [3, 4], [5, 6]])
    dtm.doc_term_matrix = csr_matrix(data)

    # Mock vectorizer
    class MockVectorizer:
        terms_list = ["term1", "term2"]

    dtm.vectorizer = MockVectorizer()

    # Set labels
    dtm.labels = ["doc1", "doc2", "doc3"]

    return dtm


@pytest.fixture
def mock_df():
    return pd.DataFrame(
        {"doc1": [10, 20, 30], "doc2": [5, 15, 25]}, index=["term1", "term2", "term3"]
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {"doc1": [10, 20, 30], "doc2": [5, 15, 25]}, index=["term1", "term2", "term3"]
    )


# Tests


def test_valid_docs_and_labels(dtm, sample_docs):
    labels = ["doc1", "doc2"]
    dtm(docs=sample_docs, labels=labels)
    assert len(dtm.docs) == len(labels)
    assert dtm.labels == labels
    assert dtm.doc_term_matrix is not None


def test_docs_with_token_lists(dtm):
    docs = [["hello", "world"], ["test", "document"]]
    labels = ["doc1", "doc2"]
    dtm(docs=docs, labels=labels)
    assert len(dtm.docs) == 2
    assert dtm.docs == docs


def test_docs_with_spacy_docs(dtm, sample_docs):
    labels = ["doc1", "doc2"]
    dtm(docs=sample_docs, labels=labels)
    assert len(dtm.docs) == 2
    assert all(isinstance(doc, list) for doc in dtm.docs)


def test_mismatched_docs_labels(dtm, sample_docs):
    labels = ["doc1"]
    with pytest.raises(
        LexosException, match="The number of docs must match the number of labels."
    ):
        dtm(docs=sample_docs, labels=labels)


def test_default_labels(dtm, sample_docs):
    dtm(docs=sample_docs, labels=None)
    assert dtm.labels == ["Doc1", "Doc2"]


def test_invalid_sorting_algorithm(dtm, sample_docs):
    dtm.alg = "invalid_algorithm"
    with pytest.raises(LexosException):
        dtm(docs=sample_docs, labels=None)


def test_vectorizer_error(dtm):
    dtm.vectorizer = None
    with pytest.raises(LexosException, match="Error building DTM"):
        dtm(docs=[["test"]], labels=["doc1"])


def test_sorted_terms_list_basic(dtm_with_terms):
    """Test basic natural sorting of terms."""
    expected = ["2term", "10term", "term1", "term2", "Term3"]
    assert dtm_with_terms.sorted_terms_list == expected


def test_sorted_terms_list_empty():
    """Test sorting with empty terms list."""
    dtm = DTM()
    with pytest.raises(LexosException):
        dtm(docs=[], labels=[])


def test_sorted_terms_list_with_different_alg(nlp):
    """Test sorting with different algorithm."""
    dtm = DTM()
    dtm(
        docs=[nlp("Term2"), nlp("term1"), nlp("term3"), nlp("10term"), nlp("2term")],
        labels=["doc1", "doc2", "doc3", "doc4", "doc5"],
    )
    dtm.alg = ns.IGNORECASE
    expected = ["2term", "10term", "term1", "Term2", "term3"]
    assert dtm.sorted_terms_list == expected


def test_sorted_terms_list_no_vectorizer(dtm_with_terms):
    """Test behavior when vectorizer is not initialized."""
    dtm_with_terms.vectorizer = None
    with pytest.raises(AttributeError):
        _ = dtm_with_terms.sorted_terms_list


def test_sorted_term_counts_basic(mock_dtm):
    """Test basic natural sorting of term counts."""
    expected = {"2term": 4, "10term": 2, "term1": 3, "term2": 5, "Term3": 7}
    assert mock_dtm.sorted_term_counts == expected


def test_sorted_term_counts_empty():
    """Test sorting with empty vocabulary."""
    dtm = DTM()

    class MockVectorizer:
        vocabulary_terms = {}

    dtm.vectorizer = MockVectorizer()
    assert dtm.sorted_term_counts == {}


def test_sorted_term_counts_different_alg(mock_dtm):
    """Test sorting with different algorithm."""
    mock_dtm.alg = ns.IGNORECASE
    expected = {"2term": 4, "10term": 2, "term1": 3, "term2": 5, "Term3": 7}
    # print(mock_dtm.sorted_term_counts)
    # assert False
    assert mock_dtm.sorted_term_counts == expected


def test_sorted_term_counts_no_vectorizer():
    """Test behavior when vectorizer is not initialized."""
    dtm = DTM()
    with pytest.raises(AttributeError):
        _ = dtm.sorted_term_counts


def test_basic_percentages(mock_df):
    dtm = DTM()
    result = dtm._get_term_percentages(mock_df, as_str=False)
    expected = pd.DataFrame(
        {"doc1": [9.524, 19.048, 28.571], "doc2": [4.762, 14.286, 23.810]},
        index=["term1", "term2", "term3"],
    ).round(3)
    pd.testing.assert_frame_equal(result.round(3), expected)


def test_zero_sum_case():
    zero_df = pd.DataFrame({"doc1": [0, 0], "doc2": [0, 0]})
    dtm = DTM()
    result = dtm._get_term_percentages(zero_df, as_str=False)
    assert (result == 0).all().all()


def test_rounding(mock_df):
    dtm = DTM()
    result = dtm._get_term_percentages(mock_df, rounding=1, as_str=False)
    assert all(len(str(x).split(".")[-1]) <= 1 for x in result.values.flatten())


def test_with_statistics(mock_df):
    dtm = DTM()
    result = dtm._get_term_percentages(
        mock_df, as_str=False, sum=True, mean=True, median=True
    )
    assert "Total" in result.columns
    assert "Mean" in result.columns
    assert "Median" in result.columns


def test_string_output(mock_df):
    dtm = DTM()
    result = dtm._get_term_percentages(mock_df, as_str="string")
    assert all(isinstance(x, str) and x.endswith("%") for x in result.values.flatten())


def test_different_rounding_values(mock_df):
    dtm = DTM()
    for rounding in [1, 2, 4]:
        result = dtm._get_term_percentages(mock_df, rounding=1, as_str=False)
        assert all(
            len(str(x).split(".")[-1]) <= rounding for x in result.values.flatten()
        )


def test_basic_conversion(mock_df_dtm):
    df = mock_df_dtm.to_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["doc1", "doc2", "doc3"]
    assert list(df.index) == ["term1", "term2"]

def test_dtm_shape_property(mock_df_dtm):
    '''Test the DTM.shape property returns the correct dimensions'''
    expected_shape = (3, 2)
    assert mock_df_dtm.shape == expected_shape

def test_sorting(mock_df_dtm):
    df = mock_df_dtm.to_df(by=["doc1"], ascending=False)
    assert df.index.tolist() == ["term2", "term1"]


def test_percentage_conversion(dtm_with_terms):
    df = dtm_with_terms.to_df(as_percent=False)
    print(df)
    df = dtm_with_terms.to_df(as_percent=True)
    assert all(isinstance(x, float) for x in df.values.flatten())


def test_percentage_conversion_string(mock_df_dtm):
    df = mock_df_dtm.to_df(as_percent="string")
    assert all(isinstance(x, str) and x.endswith("%") for x in df.values.flatten())


def test_rounding_in_get_term_percentages_method(mock_df_dtm):
    rounding = 2
    df = mock_df_dtm.to_df(as_percent=True, rounding=rounding)
    values = [str(x).split(".")[1] for x in df.values.flatten()]
    assert all(len(x) <= rounding for x in values)


def test_transpose(mock_df_dtm):
    df = mock_df_dtm.to_df(transpose=True)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["term1", "term2"]
    assert list(df.index) == ["doc1", "doc2", "doc3"]


def test_statistics(mock_df_dtm):
    df = mock_df_dtm.to_df(sum=True, mean=True, median=True)
    assert "Total" in df.columns
    assert "Mean" in df.columns
    assert "Median" in df.columns

def test_to_df_with_statistics_no_percentages(mock_df_dtm):
    '''Test to_df method with sum, mean, and median when not not converting to percantages'''
    df = mock_df_dtm.to_df(sum=True, mean=True, median=True, as_percent=False)

    # assert that the new columns are present
    assert "Total" in df.columns
    assert "Mean" in df.columns
    assert "Median" in df.columns

    # assert the shape (original columns + 3 new statistics columns)
    # original shape is (terms, docs) -> after T is (docs, terms)
    # then, after T again in to_df, it is (terms, docs)
    # so if mock_df_dtm has (2,3) (terms, docs)
    # then df.shape should be (2, 3+3) = (2+6)
    assert df.shape == (2, 3+3) # (terms, docs + stats) assuming 2 terms and 3 docs in mock_df_dtm

def test_combined_options(mock_df_dtm):
    rounding = 2
    df = mock_df_dtm.to_df(
        by=["doc1"],
        ascending=False,
        as_percent="string",
        rounding=rounding,
        transpose=True,
        sum=True,
    )
    assert isinstance(df, pd.DataFrame)
    assert "Total" in df.index
    assert all(isinstance(x, str) and x.endswith("%") for x in df.values.flatten())
    values = [x.split(".")[1] for x in df.values.flatten()]
    assert all(len(x) <= rounding + 1 for x in values)


def test_valid_sorting_algorithm(dtm):
    """Test validation with valid sorting algorithm."""
    dtm.alg = ns.IGNORECASE
    assert dtm._validate_sorting_algorithm() is True


def test_invalid_sorting_algorithm_in_validation_method(dtm):
    """Test validation with invalid sorting algorithm."""
    dtm.alg = "invalid_alg"
    with pytest.raises(LexosException) as excinfo:
        dtm._validate_sorting_algorithm()
    assert "Invalid sorting algorithm" in str(excinfo.value)
    assert "Valid algorithms for `alg` are:" in str(excinfo.value)


def test_none_sorting_algorithm(dtm):
    """Test validation with None as sorting algorithm."""
    dtm.alg = None
    with pytest.raises(LexosException) as excinfo:
        dtm._validate_sorting_algorithm()
    assert "Invalid sorting algorithm" in str(excinfo.value)


def test_error_message_formatting(dtm):
    """Test error message contains all valid algorithms."""
    dtm.alg = "invalid_alg"
    with pytest.raises(LexosException) as excinfo:
        dtm._validate_sorting_algorithm()
    for locale in ns:
        assert f"ns.{locale.name}" in str(excinfo.value)
