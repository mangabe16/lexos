"""test_ngrams.py.

Last Update: Jan 25, 2025
"""

import pytest
import spacy

from lexos.exceptions import LexosException
from lexos.tokenizer import SliceTokenizer
from lexos.tokenizer.ngrams import Ngrams

nlp = spacy.load("en_core_web_sm")

@pytest.fixture
def ng():
    """Fixture for the Ngrams class."""
    return Ngrams()

def test_ngrams_stopwords(ng):
    """Adds stopwords to the ngrams object."""
    ng.filter_stops = ["Stop", "Words"]
    assert ng.stopwords == ["Stop", "Words"]

def test_ngrams_from_doc_output(ng):
    """Generate a list of ngrams from a Doc."""
    doc = nlp("This is a test.")
    ngrams = ng.from_doc(doc, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test")]
    ngrams = ng.from_doc(doc, output="text")
    assert list(ngrams) == ["This is", "is a", "a test"]
    ngrams = ng.from_doc(doc, output="spans")
    assert [span.text for span in ngrams] == ["This is", "is a", "a test"]

def test_ngrams_from_text_output(ng):
    """Generate a list of ngrams from a list of tokens."""
    text = "This is a test."
    ngrams = ng.from_text(text, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test.")]
    ngrams = ng.from_text(text, output="text")
    assert list(ngrams) == ["This is", "is a", "a test."]

def test_ngrams_from_tokens_output(ng):
    """Generate a list of ngrams from an iterable of tokens."""
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test")]
    ngrams = ng.from_tokens(tokens, output="text")
    assert list(ngrams) == ["This is", "is a", "a test"]

def test_ngrams_from_docs(ng):
    """Generate a list of ngrams from a list of docs."""
    doc = nlp("This is a test.")
    ngrams = ng.from_docs([doc, doc], output="text")
    for doc_ng in ngrams:
        assert list(doc_ng) == ["This is", "is a", "a test"]

def test_ngrams_from_texts(ng):
    """Generate a list of ngrams from a list of texts."""
    text = "This is a test."
    ngrams = ng.from_texts([text, text], output="text")
    for doc_ng in ngrams:
        assert list(doc_ng) == ["This is", "is a", "a test."]

def test_ngrams_from_token_lists(ng):
    """Generate a list of ngrams from a list of token lists."""
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_token_lists([tokens, tokens], output="text")
    for doc_ng in ngrams:
        assert list(doc_ng) == ["This is", "is a", "a test"]

def test_ngrams_from_doc_filter_nums(ng):
    """Generate a list of ngrams excluding numbers from a doc."""
    doc = nlp("This is test ten of 10.")
    ngrams = ng.from_doc(doc, filter_nums=True, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test")]

def test_ngrams_from_doc_filter_digits(ng):
    """Generate a list of ngrams excluding digits from a doc."""
    doc = nlp("This is test ten of 10.")
    ngrams = ng.from_doc(doc, filter_digits=True, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test"), ("test", "ten"), ("ten", "of")]

def test_ngrams_from_doc_filter_punct(ng):
    """Generate a list of ngrams including punctuation from a doc."""
    doc = nlp("This is test.")
    ngrams = ng.from_doc(doc, filter_punct=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test"), ("test", ".")]

def test_ngrams_from_doc_min_freq(ng):
    """Generate a list of ngrams with that occur at least min_freq times from a doc."""
    doc = nlp("This is test.")
    ngrams = ng.from_doc(doc, min_freq=2, output="tuples")
    assert list(ngrams) == []

def test_ngrams_from_doc_filter_stops(ng):
    """Generate a list of ngrams excluding stopwords from a doc."""
    doc = nlp("This is really big test.")
    ngrams = ng.from_doc(doc, filter_stops=True, output="tuples")
    assert list(ngrams) == [("big", "test")]
    ngrams = ng.from_doc(doc, filter_stops=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "really"), ("really", "big"), ("big", "test")]

def test_ngrams_from_doc_exception(ng):
    """Generate an exception when an invalid output type is provided using from_doc."""
    doc = nlp("This test should raise an exception.")
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_doc(doc, output="invalid_format"))

def test_ngrams_from_text_filter_digits(ng):
    """Generate a list of ngrams excluding digits from a string."""
    text = "This is test ten of 10."
    ngrams = ng.from_text(text, filter_digits=True, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test"), ("test", "ten"), ("ten", "of")]

def test_ngrams_from_text_filter_punct(ng):
    """Generate a list of ngrams including punctuation from a string."""
    text = "This is test."
    ngrams = ng.from_text(text, filter_punct=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test.")]

def test_ngrams_from_text_filter_stops(ng):
    """Generate a list of ngrams excluding stopwords from a string."""
    text = "This is test."
    ngrams = ng.from_text(text, filter_stops=["is"], output="tuples")
    assert list(ngrams) == [("This", "test.")]

def test_ngrams_from_text_drop_ws():
    """This test has to be peformed with SliceTokenizer."""
    pass

def test_ngrams_from_text_min_freq(ng):
    """Generate a list of ngrams with that occur at least min_freq times from a string."""
    text = "This is test."
    ngrams = ng.from_text(text, min_freq=2, output="tuples")
    assert list(ngrams) == []

def test_ngrams_from_text_exception(ng):
    """Generate an exception when an invalid output type is provided using from_text."""
    text = "This test should raise an exception."
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_text(text, output="invalid_format"))

def test_ngrams_from_tokens_filter_digits(ng):
    """Generate a list of ngrams excluding digits from a list of tokens."""
    tokens = ["This", "is", "test", "ten", "of", "10", "."]
    ngrams = ng.from_tokens(tokens, filter_digits=True, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test"), ("test", "ten"), ("ten", "of")]

def test_ngrams_from_tokens_filter_punct(ng):
    """Generate a list of ngrams including punctuation from a list of tokens."""
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, filter_punct=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test"), ("test", ".")]

def test_ngrams_from_tokens_filter_stops(ng):
    """Generate a list of ngrams excluding stopwords from a list of tokens."""
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, filter_stops=["is"], output="tuples")
    assert list(ngrams) == [("This", "a"), ("a", "test")]

def test_ngrams_from_tokens_drop_ws(ng):
    """This test has to be peformed with SliceTokenizer."""
    pass

def test_ngrams_from_tokens_min_freq(ng):
    """Generate a list of ngrams with that occur at least min_freq times from a list of tokens."""
    tokens = ["This ", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, min_freq=2, output="tuples")
    assert list(ngrams) == []

def test_ngrams_from_text_slice_tokenizer(ng):
    """Generate a list of ngrams from a string using SliceTokenizer."""
    text = "This is test."
    tokenizer = SliceTokenizer(n=2, drop_ws=True)
    ngrams = ng.from_text(text, tokenizer=tokenizer, output="tuples")
    assert list(ngrams) == [("Th", "is"), ("is", "is"), ("is", "te"), ("te", "st")]
    tokenizer = SliceTokenizer(n=2, drop_ws=False)
    ngrams = ng.from_text(text, tokenizer=tokenizer, output="tuples")
    assert list(ngrams) == [("Th", "is"), ("is", "i"), ("i", "s"), ("s", "te"), ("te", "st")]

def test_ngrams_from_text_nlp(ng):
    """Generate a list of ngrams from a string using spaCy NLP."""
    text = "This is test."
    ngrams = ng.from_text(text, tokenizer=nlp, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test")]

def test_ngrams_from_tokens_exception(ng):
    """Generate an exception when an invalid output type is provided using from_tokens."""
    tokens = ["This", "test", "should", "raise", "an", "exception", "."]
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_tokens(tokens, output="invalid_format"))