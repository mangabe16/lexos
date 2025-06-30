"""test_kwic.py.

Unit tests for the Kwic class in lexos.kwic.

Purpose:

These tests verify the correct behavior of the Kwic class which searchs for keywords in a spaCy Doc or string and
returns all instances of the keyword with the surrounding context as a list.

The tests ensure consistent and correct output across multiple inputs, including:
    - Finding keywords in a spaCy Doc object
    - Finding keywords in a string
    - Searching for a keyword that does not exist in the text
    - Using regex patterns as keywords

Usage:
To run the test for this module:
    uv run pytest tests/kwic/kwic_test.py

Last Updated: 6/27/25
Last Tested: 6/27/25
"""

from src.lexos.kwic import Kwic
from src.lexos.tokenizer import Tokenizer
from typing import Iterable
from lexos.exceptions import LexosException
import pandas as pd
import pytest
from pydantic import ValidationError
import spacy


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Fixture of a tokenizer object."""
    return Tokenizer()

@pytest.fixture
def kwic() -> Kwic:
    """Fixture of a Kwic object."""
    return Kwic(model_name="en_core_web_sm")  # Use a specific spaCy model for testing


@pytest.fixture
def spacy_doc_sentences(tokenizer) -> str:
    """Fixture of a sample text that is tokenized into sentences."""
    text = "This is the first sentence. This is a second sentence. This third sentence has keyword in it."
    doc = tokenizer(text)
    return doc


def test_kwic_find_with_doc(tokenizer, kwic) -> None:
    """Test KWIC find method with a Doc object."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    kwic_results = kwic.find(doc=doc, keyword="test", window_size=3, pad_context=True)
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert list(results_list) == [[(" a ", "test", " st"), ("to ", "test", " th")]]


def test_kwic_find_with_string(tokenizer, kwic) -> None:
    """Test KWIC find method with a string input."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc1 = tokenizer(text)
    doc2 = tokenizer(text)
    kwic_results = kwic.find(
        doc=[doc1, doc2], keyword="test", window_size=3, pad_context=True
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert results_list[0] == [(" a ", "test", " st"), ("to ", "test", " th")]
    assert results_list[1] == [(" a ", "test", " st"), ("to ", "test", " th")]


def test_kwic_word_not_found(tokenizer, kwic) -> None:
    """Test KWIC find method with a word not found in the text."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    kwic_results = kwic.find(
        doc=doc, keyword="nonexistent", window_size=3, pad_context=True
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert results_list == [[]]  # No results expected for a word not found


def test_kwic_find_with_regex_keyword(tokenizer, kwic) -> None:
    """Test KWIC find method with a regex keyword."""
    text = "This is a test string with multiple variations of the word Test, which will be searched using a regex expression."
    doc = tokenizer(text)
    kwic_results = kwic.find(
        doc=doc, keyword=r"[Tt]est", window_size=50, pad_context=False
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert results_list == [
        [
            (
                "This is a ",
                "test",
                " string with multiple variations of the word Test,",
            ),
            (
                " test string with multiple variations of the word ",
                "Test",
                ", which will be searched using a regex expression.",
            ),
        ]
    ]


def test_kwic_find_to_dataframe(tokenizer, kwic) -> None:
    """Test KWIC find method with DataFrame output."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    kwic_df = kwic.find(
        doc=doc, keyword="test", window_size=3, pad_context=True, dataframe_format=True
    )
    assert isinstance(kwic_df, pd.DataFrame)
    assert len(kwic_df) == 2
    assert list(kwic_df.columns) == ["Left", "Keyword", "Right"]
    assert kwic_df.iloc[0].tolist() == [" a ", "test", " st"]
    assert kwic_df.iloc[1].tolist() == ["to ", "test", " th"]


def test_kwic_find_empty_string(tokenizer, kwic) -> None:
    """Test KWIC find method with an empty string."""
    text = ""
    doc = tokenizer(text)
    kwic_results = kwic.find(doc=doc, keyword="test", window_size=3, pad_context=True)
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert results_list == [[]]  # No results expected for an empty string


def test_multiple_keywords(tokenizer, kwic) -> None:
    """Test KWIC find method with multiple keywords."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    keywords = ["test", "Kwic"]
    kwic_results = kwic.find_multiple_keywords(
        doc=doc, keywords=keywords, window_size=3, pad_context=True
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert len(results_list) == 3
    assert results_list[0] == (" a ", "test", " st", "test")
    assert results_list[1] == ("to ", "test", " th", "test")
    assert results_list[2] == ("he ", "Kwic", " mo", "Kwic")


def test_multiple_keywords_none_provided(tokenizer, kwic) -> None:
    """Test KWIC find method with no keywords provided."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    kwic_results = kwic.find_multiple_keywords(
        doc=doc, keywords=[], window_size=3, pad_context=True
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert results_list == []


def test_multiple_keywords_to_dataframe(tokenizer, kwic) -> None:
    """Test KWIC find method with multiple keywords and DataFrame output."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    keywords = ["test", "Kwic"]
    kwic_df = kwic.find_multiple_keywords(
        doc=doc,
        keywords=keywords,
        window_size=3,
        pad_context=True,
        dataframe_format=True,
    )
    assert isinstance(kwic_df, pd.DataFrame)
    assert len(kwic_df) == 3
    assert list(kwic_df.columns) == ["Left", "Keyword", "Right", "Original Keyword"]
    assert kwic_df.iloc[0].tolist() == [" a ", "test", " st", "test"]
    assert kwic_df.iloc[1].tolist() == ["to ", "test", " th", "test"]
    assert kwic_df.iloc[2].tolist() == ["he ", "Kwic", " mo", "Kwic"]


def test_mixed_string_regex_keywords(tokenizer, kwic) -> None:
    """Test KWIC find method with a mix of string and regex keywords."""
    text = "This is a test string to test the Kwic module for finding correct words in context using kwic."
    doc = tokenizer(text)
    keywords = ["test", r"[Kk]wic"]
    kwic_results = kwic.find_multiple_keywords(
        doc=doc, keywords=keywords, window_size=3, pad_context=True
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert len(results_list) == 4
    assert results_list[0] == (" a ", "test", " st", "test")
    assert results_list[1] == ("to ", "test", " th", "test")
    assert results_list[2] == ("he ", "Kwic", " mo", "[Kk]wic")
    assert results_list[3] == ("ng ", "kwic", ".  ", "[Kk]wic")

def test_multiple_keywords_multiple_docs(tokenizer, kwic) -> None:
    """Test KWIC find method with multiple Doc objects and multiple keywords."""
    text1 = "This is a test string to test the Kwic module for finding correct words in context."
    text2 = "Another test string to check the Kwic functionality."
    doc1 = tokenizer(text1)
    doc2 = tokenizer(text2)
    keywords = ["test", "Kwic"]
    kwic_results = kwic.find_multiple_keywords(
        doc=[doc1, doc2], keywords=keywords, window_size=3, pad_context=True
    )
    results_list = list(kwic_results)
    print(results_list)
    assert isinstance(kwic_results, Iterable)
    assert len(results_list) == 5
    assert results_list[0] == (" a ", "test", " st", "test")
    assert results_list[1] == ("to ", "test", " th", "test")
    assert results_list[2] == ("he ", "Kwic", " mo", "Kwic")
    assert results_list[3] == ("er ", "test", " st", "test")
    assert results_list[4] == ("he ", "Kwic", " fu", "Kwic")


def test_find_in_sentences(spacy_doc_sentences, kwic) -> None:
    """Test KWIC find method in sentences."""
    doc = spacy_doc_sentences
    kwic_results = kwic.find_in_sentences(doc=doc, keyword="keyword", ignore_case=True)
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert len(results_list) == 1
    assert results_list[0] == ("This third sentence has ", "keyword", " in it.")


def test_find_in_sentences_requires_doc_object(kwic) -> None:
    """Test that find_in_sentences raises TypeError if doc is not a Doc object."""
    with pytest.raises(TypeError):
        kwic.find_in_sentences(
            doc="This is a string, not a Doc object.", keyword="keyword"
        )


def test_find_in_sentences_to_dataframe(spacy_doc_sentences, kwic) -> None:
    """Test KWIC find method in sentences with DataFrame output."""
    doc = spacy_doc_sentences
    kwic_df = kwic.find_in_sentences(
        doc=doc, keyword="keyword", ignore_case=True, dataframe_format=True
    )
    assert isinstance(kwic_df, pd.DataFrame)
    assert len(kwic_df) == 1
    assert list(kwic_df.columns) == ["Left", "Keyword", "Right"]
    assert kwic_df.iloc[0].tolist() == [
        "This third sentence has ",
        "keyword",
        " in it.",
    ]


def test_find_in_sentences_multiple_docs(tokenizer, kwic) -> None:
    """Test KWIC find method in sentences with multiple Doc objects."""
    text1 = "This is the first sentence. This is a second sentence."
    text2 = "This third sentence has keyword in it."
    doc1 = tokenizer(text1)
    doc2 = tokenizer(text2)
    kwic_results = kwic.find_in_sentences(
        doc=[doc1, doc2], keyword="sentence", ignore_case=True
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert len(results_list) == 3
    print(results_list[0])
    assert results_list[0] == ("This is the first ", "sentence", ".")
    assert results_list[1] == (
        "This is a second ",
        "sentence",
        ".",
    )
    assert results_list[2] == ("This third ", "sentence", " has keyword in it.")


def test_find_tokens(tokenizer, kwic) -> None:
    """Test KWIC find_tokens method."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    kwic_results = kwic.find_tokens(
        doc=doc, keyword="test", token_window=5, ignore_case=True
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert len(results_list) == 2
    assert results_list[0] == ("This is a", "test", "string to test the Kwic")
    assert results_list[1] == (
        "is a test string to",
        "test",
        "the Kwic module for finding",
    )


def test_find_tokens_no_ignore_case(tokenizer, kwic) -> None:
    """Test KWIC find_tokens method while not ignoring case."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    kwic_results = kwic.find_tokens(
        doc=doc, keyword="test", token_window=5, ignore_case=False
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert len(results_list) == 2
    assert results_list[0] == ("This is a", "test", "string to test the Kwic")
    assert results_list[1] == (
        "is a test string to",
        "test",
        "the Kwic module for finding",
    )


def test_find_tokens_no_matches(tokenizer, kwic) -> None:
    """Test KWIC find_tokens method."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    with pytest.raises(LexosException):
        kwic.find_tokens(
            doc=doc, keyword="bingo", token_window=5, ignore_case=True
        )


def test_find_tokens_to_dataframe(tokenizer, kwic) -> None:
    """Test KWIC find_tokens method with DataFrame output."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    kwic_df = kwic.find_tokens(
        doc=doc,
        keyword="test",
        token_window=5,
        ignore_case=True,
        dataframe_format=True,
    )
    assert isinstance(kwic_df, pd.DataFrame)
    assert len(kwic_df) == 2
    assert list(kwic_df.columns) == ["Left", "Keyword", "Right"]
    assert kwic_df.iloc[0].tolist() == ["This is a", "test", "string to test the Kwic"]
    assert kwic_df.iloc[1].tolist() == [
        "is a test string to",
        "test",
        "the Kwic module for finding",
    ]


def test_find_tokens_multiple_sentences(tokenizer, kwic) -> None:
    """Test KWIC find_tokens method with multiple sentences."""
    text = "This is the first sentence. This is a second sentence. This third sentence has keyword in it."
    doc = tokenizer(text)
    kwic_results = kwic.find_tokens(
        doc=doc, keyword="sentence", token_window=5, ignore_case=True
    )
    results_list = list(kwic_results)
    print(results_list[2])
    assert isinstance(kwic_results, Iterable)
    assert len(results_list) == 3
    assert results_list[0] == ("This is the first", "sentence", ". This is a second")
    assert results_list[1] == (
        ". This is a second",
        "sentence",
        ". This third sentence has",
    )
    assert results_list[2] == (
        "second sentence. This third",
        "sentence",
        "has keyword in it.",
    )


def test_find_tokens_empty_string(tokenizer, kwic) -> None:
    """Test KWIC find_tokens method with an empty string."""
    text = ""
    doc = tokenizer(text)
    with pytest.raises(LexosException):
        kwic.find_tokens(
            doc=doc, keyword="test", token_window=5, ignore_case=True
        )


def test_find_tokens_invalid_keyword(tokenizer, kwic) -> None:
    """Test KWIC find_tokens method with an invalid keyword."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    with pytest.raises(ValidationError):
        kwic.find_tokens(
            doc=doc, keyword=123, token_window=5, ignore_case=True
        )  # Keyword must be a string


def test_find_tokens_multiple_docs(tokenizer, kwic) -> None:
    """Test KWIC find_tokens method with multiple Doc objects."""
    text1 = "This is a test string to test the Kwic module for finding correct words in context."
    text2 = "Another test string to check the Kwic functionality."
    doc1 = tokenizer(text1)
    doc2 = tokenizer(text2)
    kwic_results = kwic.find_tokens(
        doc=[doc1, doc2], keyword="test", token_window=5, ignore_case=True
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert len(results_list) == 3
    assert results_list[0] == ("This is a", "test", "string to test the Kwic")
    assert results_list[1] == (
        "is a test string to",
        "test",
        "the Kwic module for finding",
    )
    assert results_list[2] == ("Another", "test", "string to check the Kwic")
