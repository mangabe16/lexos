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

Last Updated: 6/18/25
Last Tested: 6/18/25
"""

from src.lexos.kwic import Kwic
from src.lexos.tokenizer import Tokenizer
from typing import Iterable
import pytest


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Fixture of a tokenizer object."""
    return Tokenizer()


def test_kwic_find_with_doc(tokenizer) -> None:
    """Test KWIC find method with a Doc object."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    kwic_results = Kwic.find(doc=doc, keyword="test", window_size=3, pad_context=True)
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert results_list == [(" a ", "test", " st"), ("to ", "test", " th")]


def test_kwic_find_with_string(tokenizer) -> None:
    """Test KWIC find method with a string input."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    kwic_results = Kwic.find(doc=text, keyword="test", window_size=3, pad_context=True)
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert results_list == [(" a ", "test", " st"), ("to ", "test", " th")]


def test_kwic_word_not_found(tokenizer) -> None:
    """Test KWIC find method with a word not found in the text."""
    text = "This is a test string to test the Kwic module for finding correct words in context."
    doc = tokenizer(text)
    kwic_results = Kwic.find(
        doc=doc, keyword="nonexistent", window_size=3, pad_context=True
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert results_list == []  # No results expected for a word not found


def test_kwic_find_with_regex_keyword(tokenizer) -> None:
    """Test KWIC find method with a regex keyword."""
    text = "This is a test string with multiple variations of the word Test, which will be searched using a regex expression."
    doc = tokenizer(text)
    kwic_results = Kwic.find(
        doc=doc, keyword=r"[Tt]est", window_size=50, pad_context=False
    )
    results_list = list(kwic_results)
    assert isinstance(kwic_results, Iterable)
    assert results_list == [
        ("This is a ", "test", " string with multiple variations of the word Test,"),
        (
            " test string with multiple variations of the word ",
            "Test",
            ", which will be searched using a regex expression.",
        ),
    ]
