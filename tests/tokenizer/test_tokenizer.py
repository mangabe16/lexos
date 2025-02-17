"""test_tokenizer.py.

Last Update: Jan 24, 2025
"""

from typing import Generator

import pytest
from spacy.tokens import Doc, Token

from src.lexos.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()


def test_add_extension(tokenizer):
    tokenizer.add_extension("test_ext", default="default_value")
    assert Token.has_extension("test_ext")


def test_add_stopwords(tokenizer):
    stopwords = ["the", "a", "an"]
    tokenizer.add_stopwords(stopwords)
    for term in stopwords:
        assert tokenizer.nlp.vocab[term].is_stop
        assert term in tokenizer.stopwords


def test_call(tokenizer):
    doc = tokenizer("This is a test.")
    assert isinstance(doc, Doc)
    assert doc.text == "This is a test."


def test_call_multiple_texts(tokenizer):
    docs = list(tokenizer(["This is a test.", "This is another test."]))
    assert isinstance(docs[0], Doc)
    assert docs[0].text == "This is a test."
    assert isinstance(docs[1], Doc)
    assert docs[1].text == "This is another test."


def test_make_doc(tokenizer):
    doc = tokenizer.make_doc("This is a test.")
    assert isinstance(doc, Doc)
    assert doc.text == "This is a test."


def test_make_docs(tokenizer):
    texts = ["This is a test.", "Another test."]
    docs = list(tokenizer.make_docs(texts))
    assert all(isinstance(doc, Doc) for doc in docs)
    assert [doc.text for doc in docs] == texts


def test_remove_extension(tokenizer):
    tokenizer.add_extension("test_ext", default="default_value")
    tokenizer.remove_extension("test_ext")
    assert not Token.has_extension("test_ext")


def test_remove_stopwords(tokenizer):
    stopwords = ["the", "a", "an"]
    tokenizer.add_stopwords(stopwords)
    assert tokenizer.nlp.vocab["the"].is_stop
    tokenizer.remove_stopwords("the")
    assert not tokenizer.nlp.vocab["the"].is_stop
    assert "the" not in tokenizer.stopwords
