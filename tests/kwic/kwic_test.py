"""test_kwic.py.

Last Updated: February 18, 2025
Last Tested: February 18, 2025.
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
