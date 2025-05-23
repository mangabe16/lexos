"""test_tokenizer.py.

Last Update: Jan 24, 2025
"""

from typing import Generator

import pytest
from spacy.tokens import Doc, Token

from lexos.exceptions import LexosException
from src.lexos.tokenizer import Tokenizer
from src.lexos.tokenizer import SliceTokenizer
from src.lexos.tokenizer import WhitespaceTokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()

@pytest.fixture
def sliceTokenizer():
    return SliceTokenizer(n = 4)

@pytest.fixture
def whitespaceTokenizer():
    return WhitespaceTokenizer()

def test_incorrect_model_exception():
    with pytest.raises(LexosException, match=f"Error loading model non_existent_model. Please check the name and try again. You may need to install the model on your system."):
        tokenizer = Tokenizer(model="non_existent_model")


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

@pytest.mark.xfail(reason="Type hinting is making this test difficult")
def test_call_incorrect_iterable(tokenizer):
    with pytest.raises(LexosException, match="Input must be a string or an iterable of strings."):
        doc = tokenizer(["yabadaba", 123])


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
    doc = tokenizer.make_doc("This is another test.", max_length=40, disable=["senter"])
    assert isinstance(doc, Doc)
    assert tokenizer.max_length == 40
    assert "senter" in tokenizer.nlp.disabled


def test_make_docs(tokenizer):
    texts = ["This is a test.", "Another test."]
    docs = list(tokenizer.make_docs(texts))
    assert all(isinstance(doc, Doc) for doc in docs)
    assert [doc.text for doc in docs] == texts
    texts = ["This is a another test.", "Another another test."]
    docs = list(tokenizer.make_docs(texts, max_length=200, disable=["senter"]))
    assert all(isinstance(doc, Doc) for doc in docs)
    assert [doc.text for doc in docs] == texts
    assert tokenizer.max_length == 200
    assert "senter" in tokenizer.nlp.disabled


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

def test_pipeline(tokenizer):
    pipeline = tokenizer.pipeline
    assert pipeline == ["senter"]
    
def test_components(tokenizer):
    components = tokenizer.components
    assert isinstance(components, list)
    assert len(components) == 1
    assert components[0][0] == "senter"

def test_disabled(tokenizer):
    disabled = tokenizer.disabled
    assert disabled == []

def test_slice_tokenizer(sliceTokenizer):
    text = "This is a test."
    slices = sliceTokenizer(text)
    # Default n=4, drop_ws=True, so spaces are removed: 'Thisisatest.'
    # Slices: ['This', 'isat', 'est.']
    assert slices == ["This", "isat", "est."]

def test_white_space_tokenizer(whitespaceTokenizer):
    text = "This is a test."
    tokens = whitespaceTokenizer(text)
    assert list(tokens) == ["This", "is", "a", "test."]
