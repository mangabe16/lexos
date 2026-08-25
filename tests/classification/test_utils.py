"""test_utils.py.

Coverage: 100%
Last update: August 25, 2026
Last test: August 25, 2026
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from lexos.classification.utils import (
    PredictionSaver,
    _to_dense,
    _tokenize_items,
    save_predictions,
)


class FakeToken:
    """Minimal token object matching the attributes used by the utility."""

    def __init__(self, text: str, is_alpha: bool = True):
        """Initialize a token with text and alphabetic-token metadata."""
        self.text = text
        self.is_alpha = is_alpha


class FakeDocument:
    """Minimal document object matching the utility's document protocol."""

    def __init__(self, *tokens: FakeToken):
        """Initialize a document from an ordered collection of fake tokens."""
        self.text = " ".join(token.text for token in tokens)
        self._tokens = tokens

    def __iter__(self):
        """Iterate over the document's fake tokens."""
        return iter(self._tokens)


def test_tokenize_items_returns_unigrams_and_bigrams():
    """Tokenization includes normalized unigrams and adjacent bigrams when enabled."""
    assert _tokenize_items(["Alpha beta gamma"], include_bigrams=True) == [
        ["alpha", "beta", "gamma", "alpha_beta", "beta_gamma"]
    ]


def test_tokenize_items_supports_document_like_items():
    """Tokenization accepts document-like objects with token metadata."""
    document = FakeDocument(FakeToken("Alpha"), FakeToken("beta"))

    assert _tokenize_items([document], include_bigrams=True) == [
        ["alpha", "beta", "alpha_beta"]
    ]


def test_tokenize_items_can_exclude_bigrams():
    """Tokenization can be limited to normalized unigrams."""
    assert _tokenize_items(["Alpha beta"], include_bigrams=False) == [["alpha", "beta"]]


def test_tokenize_items_supports_iterable_tokens():
    """Tokenization accepts non-string token iterables."""
    assert _tokenize_items([("Alpha", "beta")], include_bigrams=False) == [
        ["alpha", "beta"]
    ]


def test_tokenize_items_rejects_empty_documents():
    """Tokenization rejects documents that produce no alphabetic tokens."""
    with pytest.raises(ValueError, match="At least one document produced zero tokens"):
        _tokenize_items(["123 !!!"])


def test_tokenize_items_accepts_empty_input():
    """Tokenization returns an empty result for an empty input sequence."""
    assert _tokenize_items([]) == []


def test_to_dense_converts_sparse_matrix():
    """Sparse matrices are converted to dense NumPy arrays."""
    result = _to_dense(csr_matrix([[1, 0], [0, 2]]))

    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 2]]))
    assert isinstance(result, np.ndarray)


def test_to_dense_preserves_dense_array_values():
    """Dense inputs are converted to NumPy arrays without changing values."""
    result = _to_dense([[1, 2], [3, 4]])

    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))


def test_save_predictions_writes_csv(tmp_path: Path):
    """Prediction rows are written with the expected CSV columns."""
    output_file = tmp_path / "predictions.csv"

    save_predictions(["doc-1", "doc-2"], ["a", "b"], str(output_file))

    result = pd.read_csv(output_file)
    expected = pd.DataFrame({"filename": ["doc-1", "doc-2"], "prediction": ["a", "b"]})
    pd.testing.assert_frame_equal(result, expected)


def test_prediction_saver_uses_default_output_path(tmp_path: Path):
    """PredictionSaver writes to its configured default path."""
    output_file = tmp_path / "saved.csv"
    saver = PredictionSaver(default_output=str(output_file))

    saver.save(("doc-1", "doc-2"), ("a", "b"))

    result = pd.read_csv(output_file)
    assert result.to_dict(orient="records") == [
        {"filename": "doc-1", "prediction": "a"},
        {"filename": "doc-2", "prediction": "b"},
    ]


def test_prediction_saver_overrides_default_output_path(tmp_path: Path):
    """PredictionSaver accepts a per-call output path override."""
    default_file = tmp_path / "default.csv"
    override_file = tmp_path / "override.csv"
    saver = PredictionSaver(default_output=str(default_file))

    saver.save(["doc-1"], ["a"], output_file=str(override_file))

    assert override_file.exists()
    assert not default_file.exists()
