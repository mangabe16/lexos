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
    return Ngrams()

def test_ngrams_from_doc_output(ng):
    doc = nlp("This is a test.")
    ngrams = ng.from_doc(doc, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test")]
    ngrams = ng.from_doc(doc, output="text")
    assert list(ngrams) == ["This is", "is a", "a test"]
    ngrams = ng.from_doc(doc, output="spans")
    assert [span.text for span in ngrams] == ["This is", "is a", "a test"]

def test_ngrams_from_text_output(ng):
    text = "This is a test."
    ngrams = ng.from_text(text, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test.")]
    ngrams = ng.from_text(text, output="text")
    assert list(ngrams) == ["This is", "is a", "a test."]

def test_ngrams_from_tokens_output(ng):
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test")]
    ngrams = ng.from_tokens(tokens, output="text")
    assert list(ngrams) == ["This is", "is a", "a test"]

def test_ngrams_from_docs(ng):
    doc = nlp("This is a test.")
    ngrams = ng.from_docs([doc, doc], output="text")
    for doc_ng in ngrams:
        assert list(doc_ng) == ["This is", "is a", "a test"]

def test_ngrams_from_texts(ng):
    text = "This is a test."
    ngrams = ng.from_texts([text, text], output="text")
    for doc_ng in ngrams:
        assert list(doc_ng) == ["This is", "is a", "a test."]

def test_ngrams_from_token_lists(ng):
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_token_lists([tokens, tokens], output="text")
    for doc_ng in ngrams:
        assert list(doc_ng) == ["This is", "is a", "a test"]

def test_ngrams_from_doc_filter_nums(ng):
    doc = nlp("This is test ten of 10.")
    ngrams = ng.from_doc(doc, filter_nums=True, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test")]

def test_ngrams_from_doc_filter_digits(ng):
    doc = nlp("This is test ten of 10.")
    ngrams = ng.from_doc(doc, filter_digits=True, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test"), ("test", "ten"), ("ten", "of")]

def test_ngrams_from_doc_filter_punct(ng):
    doc = nlp("This is test.")
    ngrams = ng.from_doc(doc, filter_punct=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test"), ("test", ".")]

def test_ngrams_from_doc_min_freq(ng):
    doc = nlp("This is test.")
    ngrams = ng.from_doc(doc, min_freq=2, output="tuples")
    assert list(ngrams) == []

def test_ngrams_from_doc_filter_stops(ng):
    doc = nlp("This is really big test.")
    ngrams = ng.from_doc(doc, filter_stops=True, output="tuples")
    assert list(ngrams) == [("big", "test")]
    ngrams = ng.from_doc(doc, filter_stops=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "really"), ("really", "big"), ("big", "test")]

def test_ngrams_from_doc_exception(ng):
    doc = nlp("This test should raise an exception.")
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_doc(doc, output="invalid_format"))

def test_ngrams_from_text_filter_digits(ng):
    text = "This is test ten of 10."
    ngrams = ng.from_text(text, filter_digits=True, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test"), ("test", "ten"), ("ten", "of")]

def test_ngrams_from_text_filter_punct(ng):
    text = "This is test."
    ngrams = ng.from_text(text, filter_punct=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test.")]

def test_ngrams_from_text_filter_stops(ng):
    text = "This is test."
    ngrams = ng.from_text(text, filter_stops=["is"], output="tuples")
    assert list(ngrams) == [("This", "test.")]

def test_ngrams_from_text_drop_ws():
    """This test has to be peformed with SliceTokenizer."""
    pass

def test_ngrams_from_text_min_freq(ng):
    text = "This is test."
    ngrams = ng.from_text(text, min_freq=2, output="tuples")
    assert list(ngrams) == []

def test_ngrams_from_text_exception(ng):
    text = "This test should raise an exception."
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_text(text, output="invalid_format"))

def test_ngrams_from_tokens_filter_digits(ng):
    text = "This is test ten of 10."
    ngrams = ng.from_text(text, filter_digits=True, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test"), ("test", "ten"), ("ten", "of")]

def test_ngrams_from_tokens_filter_punct(ng):
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, filter_punct=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test"), ("test", ".")]

def test_ngrams_from_tokens_filter_stops(ng):
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, filter_stops=["is"], output="tuples")
    assert list(ngrams) == [("This", "a"), ("a", "test")]

def test_ngrams_from_tokens_drop_ws():
    """This test has to be peformed with SliceTokenizer."""
    pass

def test_ngrams_from_tokens_min_freq(ng):
    tokens = ["This ", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, min_freq=2, output="tuples")
    assert list(ngrams) == []

def test_ngrams_from_text_slice_tokenizer(ng):
    text = "This is test."
    tokenizer = SliceTokenizer(n=2, drop_ws=True)
    ngrams = ng.from_text(text, tokenizer=tokenizer, output="tuples")
    assert list(ngrams) == [("Th", "is"), ("is", "is"), ("is", "te"), ("te", "st")]
    tokenizer = SliceTokenizer(n=2, drop_ws=False)
    ngrams = ng.from_text(text, tokenizer=tokenizer, output="tuples")
    assert list(ngrams) == [("Th", "is"), ("is", "i"), ("i", "s"), ("s", "te"), ("te", "st")]

def test_ngrams_from_text_nlp(ng):
    text = "This is test."
    ngrams = ng.from_text(text, tokenizer=nlp, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test")]

def test_ngrams_from_tokens_exception(ng):
    tokens = ["This", "test", "should", "raise", "an", "exception", "."]
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_tokens(tokens, output="invalid_format"))