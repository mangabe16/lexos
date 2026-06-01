"""test_mlp_pipeline.py.

Coverage:96%
Last update: 04/21
Last test: 05/27
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from lexos.classification import mlp_pipeline
from lexos.classification.mlp_pipeline import (
	MLPPipelineConfig,
	MLPPipelineResult,
	_apply_smote,
	_to_dense,
	_tokenize_items,
	run_mlp_authorship_pipeline,
	save_mlp_unknown_predictions,
)


def _build_tiny_authorship_dataset() -> tuple[list[str], list[str], list[str], list[str]]:
	"""Build a tiny but separable dataset for fast pipeline tests."""
	hamilton_docs = [
		"commerce treasury union federal debt",
		"federal treasury commerce credit credit",
		"union debt federal commerce treasury",
		"credit debt treasury federal union",
		"federal union commerce national treasury",
		"debt finance federal treasury commerce",
	]
	madison_docs = [
		"republic liberty faction rights people",
		"rights liberty people republic faction",
		"faction republic liberty rights rights",
		"constitution liberty republic people rights",
		"people republic faction liberty state",
		"rights people liberty constitution republic",
	]

	train_data = hamilton_docs + madison_docs
	train_labels = ["HAMILTON"] * len(hamilton_docs) + ["MADISON"] * len(madison_docs)

	unknown_data = [
		"federal treasury debt union",  # hamilton-like
		"liberty rights republic people",  # madison-like
	]
	unknown_ids = ["u0.txt", "u1.txt"]
	return train_data, train_labels, unknown_data, unknown_ids


def test_tokenize_items_accepts_raw_text_and_token_lists():
	"""Tokenize mixed raw/token inputs and verify alpha filtering with bigram output."""
	items = ["Alpha beta gamma", ["Delta", "42", "epsilon"]]

	token_lists = _tokenize_items(items, include_bigrams=True)

	assert len(token_lists) == 2
	assert all(isinstance(tokens, list) for tokens in token_lists)
	assert "alpha" in token_lists[0]
	assert "beta" in token_lists[0]
	assert "gamma" in token_lists[0]
	assert any("_" in token for token in token_lists[0])
	assert "delta" in token_lists[1]
	assert "epsilon" in token_lists[1]
	assert "42" not in token_lists[1]


def test_tokenize_items_raises_when_any_document_has_zero_tokens():
	"""Raise a clear error when any document yields no alphabetic tokens."""
	with pytest.raises(ValueError, match="produced zero tokens"):
		_tokenize_items(["1234 !!!", "alpha beta"])


def test_to_dense_uses_toarray_when_available():
	"""Use an object's toarray method when present to produce a NumPy array."""
	class FakeSparse:
		def __init__(self, arr):
			self._arr = np.asarray(arr)

		def toarray(self):
			return self._arr

	dense = _to_dense(FakeSparse([[1, 2], [3, 4]]))
	assert isinstance(dense, np.ndarray)
	assert dense.shape == (2, 2)
	assert np.array_equal(dense, np.array([[1, 2], [3, 4]]))

# needs refactoring to cover line 99
def test_apply_smote_returns_dense_when_disabled():
	"""Return dense features and unchanged labels when SMOTE is disabled."""
	x = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
	y = np.array(["A", "B"])

	x_out, y_out = _apply_smote(x, y, seed=7, enabled=False)

	assert isinstance(x_out, np.ndarray)
	assert np.array_equal(x_out, x.toarray())
	assert np.array_equal(y_out, y)


def test_apply_smote_raises_if_enabled_and_dependency_missing(monkeypatch):
	"""Raise ImportError when SMOTE is requested but imbalanced-learn is unavailable."""
	monkeypatch.setattr(mlp_pipeline, "SMOTE", None)

	with pytest.raises(ImportError, match="imbalanced-learn"):
		_apply_smote(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array(["A", "B"]), seed=1, enabled=True)


def test_run_pipeline_validates_input_lengths():
	"""Validate the main pipeline rejects empty or mismatched input lengths."""
	with pytest.raises(ValueError, match="must contain at least one sample"):
		run_mlp_authorship_pipeline(train_data=[], train_labels=[])

	with pytest.raises(ValueError, match="matching lengths"):
		run_mlp_authorship_pipeline(train_data=["a", "b"], train_labels=["A"])

	with pytest.raises(ValueError, match="matching lengths"):
		run_mlp_authorship_pipeline(
			train_data=["alpha beta", "gamma delta"],
			train_labels=["A", "B"],
			test_data=["x", "y"],
			test_ids=["only_one"],
		)


def test_run_pipeline_returns_expected_structures_without_smote():
	"""Run a lightweight end-to-end pipeline and verify result container structure."""
	train_data, train_labels, unknown_data, unknown_ids = _build_tiny_authorship_dataset()
	cfg = MLPPipelineConfig(
		seed=11,
		min_df=1,
		test_size=0.25,
		cv_splits=3,
		include_bigrams=True,
		use_smote=False,
		mlp_kwargs={
			"hidden_layer_sizes": (16,),
			"activation": "relu",
			"solver": "adam",
			"alpha": 1e-4,
			"learning_rate_init": 1e-3,
			"max_iter": 80,
		},
	)

	results = run_mlp_authorship_pipeline(
		train_data=train_data,
		train_labels=train_labels,
		test_data=unknown_data,
		test_ids=unknown_ids,
		config=cfg,
	)

	assert isinstance(results, MLPPipelineResult)

	assert set(results.holdout_metrics.keys()) == {"accuracy", "balanced_accuracy", "macro_f1"}
	assert all(0.0 <= metric <= 1.0 for metric in results.holdout_metrics.values())

	assert isinstance(results.holdout_report, pd.DataFrame)
	assert not results.holdout_report.empty

	assert isinstance(results.holdout_confusion_matrix, pd.DataFrame)
	assert results.holdout_confusion_matrix.shape == (2, 2)

	assert isinstance(results.cv_fold_metrics, pd.DataFrame)
	assert len(results.cv_fold_metrics) == cfg.cv_splits
	assert set(results.cv_fold_metrics.columns) == {"fold", "accuracy", "balanced_accuracy", "macro_f1"}

	assert set(results.cv_mean_metrics.keys()) == {"accuracy", "balanced_accuracy", "macro_f1"}

	assert isinstance(results.test_predictions, pd.DataFrame)
	assert len(results.test_predictions) == len(unknown_data)
	assert list(results.test_predictions["sample_id"]) == unknown_ids
	assert "predicted_label" in results.test_predictions.columns


def test_save_mlp_unknown_predictions_writes_csv(tmp_path):
	"""Write unknown predictions to CSV and verify saved columns and row count."""
	dummy_results = MLPPipelineResult(
		holdout_report=pd.DataFrame(),
		holdout_confusion_matrix=pd.DataFrame(),
		holdout_metrics={"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0},
		cv_fold_metrics=pd.DataFrame(),
		cv_mean_metrics={"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0},
		test_predictions=pd.DataFrame(
			{
				"sample_id": ["u0.txt", "u1.txt"],
				"predicted_label": ["HAMILTON", "MADISON"],
			}
		),
		final_model=object(),
		final_dtm=object(),
		final_scaler=object(),
	)

	output_csv = tmp_path / "predictions" / "unknown_predictions.csv"
	saved = save_mlp_unknown_predictions(dummy_results, output_csv)

	assert isinstance(saved, Path)
	assert saved.exists()

	loaded = pd.read_csv(saved)
	assert list(loaded.columns) == ["sample_id", "predicted_label"]
	assert len(loaded) == 2