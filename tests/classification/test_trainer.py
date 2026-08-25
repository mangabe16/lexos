"""test_trainer.py.

Coverage: 100%
Last update: August 25, 2026
Last test: August 25, 2026
"""

import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import random
from pydantic import Field, PrivateAttr
from typing import Sequence, Any
from collections import Counter

from lexos.classification.trainer import Classifier, Pipeline


class RecordingModel:
    """Dummy prediction model."""

    def __init__(self, predictions: Sequence[Any]):
        """Initializes RecordingModel with preconfigured prediction outputs."""
        self.predictions = np.asarray(predictions)
        self.last_input = None

    def predict(self, data):
        """Simulate model prediction by recording input matrix and returning sliced predictions."""
        self.last_input = data
        return np.asarray(self.predictions[: len(data)])


class OffsetScaler:
    """Dummy scaler for testing."""

    def __init__(self, offset: float = 1.0):
        """Initializes OffsetScaler with a fixed numerical offset."""
        self.offset = offset
        self.transform_calls = 0
        self.last_input = None

    def transform(self, data):
        """Simulate feature scoring by adding a fixed offset to input data while tracking call state."""
        self.transform_calls += 1
        self.last_input = np.asarray(data)
        return np.asarray(data) + self.offset


class DummyPipeline(Pipeline):
    """Dummy Pipeline Class to keep the test suite Pipeline agnostic."""

    discovered_features: list[str] = Field(default_factory=list)
    training_payload: dict[str, Any] = Field(default_factory=dict)
    feature_removal: str | None = None
    seed: int = 7
    include_bigrams: bool = False

    _discover_calls: int = PrivateAttr(default=0)
    _execute_calls: int = PrivateAttr(default=0)
    _last_active_features: list[str] | None = PrivateAttr(default=None)

    def discover_features(self, train_data):
        """Dummy feature discovery that justs increment counter and returns precalculated list of features."""
        self._discover_calls += 1
        return list(self.discovered_features)

    def execute_training(self, train_data, labels, active_features=None):
        """Dummy training that increases count of executions and returns preset dictionary for training_payload."""
        self._execute_calls += 1
        self._last_active_features = (
            None if active_features is None else list(active_features)
        )
        return self.training_payload

    def predict(self, data, results_payload, active_features=None):
        """Dummy prediction implementation required by the Pipeline contract."""
        if isinstance(data, pd.DataFrame):
            if hasattr(self, "_filter_active_features"):
                data = self._filter_active_features(
                    baseline_features=self.discovered_features,
                    matrix=data,
                    active_features=active_features,
                )
            else:
                data = data.loc[:, list(active_features)].to_numpy()
        elif isinstance(data, list) and data and isinstance(data[0], str):
            from lexos.classification.utils import _tokenize_items

            dtm_object = results_payload["final_dtm"]
            token_lists = _tokenize_items(
                data,
                include_bigrams=self.include_bigrams,
            )
            data = dtm_object.vectorizer.transform(token_lists)

        scaler = results_payload.get("final_scaler")
        if scaler is not None:
            data = scaler.transform(data)

        return results_payload["final_model"].predict(data)


class FilteringPipeline(DummyPipeline):
    """Dummy pipeline that records feature filtering calls."""

    _filter_calls: int = PrivateAttr(default=0)
    _last_filter_args: dict[str, Any] | None = PrivateAttr(default=None)

    def _filter_active_features(self, baseline_features, matrix, active_features):
        self._filter_calls += 1
        self._last_filter_args = {
            "baseline_features": list(baseline_features),
            "active_features": list(active_features)
            if active_features is not None
            else None,
            "matrix_shape": matrix.shape,
        }
        return matrix[active_features].to_numpy()


class TrackingClassifier(Classifier):
    """Classifier subclass that records initialize hook invocations."""

    _initialize_calls: int = PrivateAttr(default=0)

    def _initialize_model(self) -> None:
        self._initialize_calls += 1


class RecordingVectorizer:
    """Dummy vectorizer that records tokenized input."""

    def __init__(self):
        """Assgins the last_input to be None."""
        self.last_input = None

    def transform(self, data):
        """Simulate feature vectorization by returning a 2-column matrix derived from token indices."""
        self.last_input = data
        return np.asarray(
            [[float(index), float(index) + 0.5] for index, _ in enumerate(data)]
        )


class DummyDTM:
    """Minimal DTM-like object for predict() text path coverage."""

    def __init__(self):
        """Assigns internal vectorizer to be a RecordingVectorizer."""
        self.vectorizer = RecordingVectorizer()


def build_payload(
    model: RecordingModel,
    *,
    scaler: OffsetScaler | None = None,
    final_dtm: Any | None = None,
    holdout_metrics: dict[str, float] | None = None,
    holdout_report: pd.DataFrame | None = None,
    cv_mean_metrics: dict[str, float] | None = None,
    final_metrics: dict[str, float] | None = None,
    final_report: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Helper to create a dictionary of training and evaluation results payload."""
    return {
        "final_model": model,
        "final_scaler": scaler,
        "final_dtm": final_dtm,
        "holdout_metrics": holdout_metrics or {},
        "holdout_report": holdout_report
        if holdout_report is not None
        else pd.DataFrame(),
        "holdout_confusion_matrix": pd.DataFrame(),
        "cv_fold_metrics": pd.DataFrame(),
        "cv_mean_metrics": cv_mean_metrics or {},
        "final_metrics": final_metrics or {},
        "final_report": final_report if final_report is not None else pd.DataFrame(),
    }


def make_classifier(
    *,
    train_data: Any,
    labels: list[Any],
    payload: dict[str, Any],
    discovered_features: list[str] | None = None,
    features: list[str] | None = None,
    feature_removal: str | None = None,
    seed: int = 7,
    include_bigrams: bool = False,
) -> tuple[Classifier, DummyPipeline]:
    """Creates a Classifier using a DummyPipeline."""
    pipeline = DummyPipeline(
        discovered_features=discovered_features or [],
        training_payload=payload,
        feature_removal=feature_removal,
        seed=seed,
        include_bigrams=include_bigrams,
    )
    classifier = Classifier(
        train_data=train_data,
        labels=labels,
        pipeline=pipeline,
        features=features,
    )
    return classifier, pipeline


def test_validate_train_data_accepts_dataframe_and_sequences():
    """Tests Classifier class accepts and returns valid data types."""
    train_data_cases = [
        pd.DataFrame({"text": ["a", "b"]}),
        ["a", "b"],
        ("a", "b"),
    ]

    for train_data in train_data_cases:
        classifier, _ = make_classifier(
            train_data=train_data,
            labels=["x", "y"],
            payload=build_payload(RecordingModel(["x", "y"])),
        )
        if isinstance(train_data, pd.DataFrame):
            pd.testing.assert_frame_equal(classifier.train_data, train_data)
        else:
            assert list(classifier.train_data) == list(train_data)


@pytest.mark.parametrize("bad_train_data", [123, "abc", {"a": 1}])
def test_validate_train_data_rejects_invalid_types(bad_train_data):
    """Tests appropriate TypeError is raised for invalid input types."""
    with pytest.raises(TypeError):
        make_classifier(
            train_data=bad_train_data,
            labels=["x"],
            payload=build_payload(RecordingModel(["x"])),
        )


def test_pipeline_abstract_hooks_can_be_invoked_on_concrete_strategy_instance():
    """Tests calling abstract Pipeline hooks directly returns None and bypasses subclass logic."""
    pipeline = DummyPipeline(
        discovered_features=["alpha"],
        training_payload=build_payload(RecordingModel(["x"])),
    )

    assert Pipeline.discover_features(pipeline, ["a", "b"]) is None
    assert Pipeline.execute_training(pipeline, ["a", "b"], ["x", "y"]) is None
    assert pipeline._discover_calls == 0
    assert pipeline._execute_calls == 0


def test_pipeline_predict_abstract_hook_can_be_invoked_on_concrete_strategy_instance():
    """Tests calling the abstract prediction hook directly returns None."""
    pipeline = DummyPipeline(
        discovered_features=["alpha"],
        training_payload=build_payload(RecordingModel(["x"])),
    )

    assert Pipeline.predict(pipeline, ["a"], {}) is None


def test_preprocess_data_raises_on_mismatched_lengths():
    """Tests a ValueError is raised for mismatched train_data and labels lengths."""
    classifier, _ = make_classifier(
        train_data=["a", "b"],
        labels=["only_one"],
        payload=build_payload(RecordingModel(["x", "y"])),
    )

    with pytest.raises(ValueError):
        classifier._preprocess_data()


def test_preprocess_data_discovers_features_when_missing():
    """Verify _preprocess_data triggers feature discovery when no features are provided."""
    classifier, pipeline = make_classifier(
        train_data=["a", "b"],
        labels=["x", "y"],
        payload=build_payload(RecordingModel(["x", "y"])),
        discovered_features=["alpha", "beta"],
    )

    classifier._preprocess_data()

    assert classifier.features == ["alpha", "beta"]
    assert pipeline._discover_calls == 1


def test_fit_updates_internal_state_and_properties():
    """Tests internal properties are properly updated by fit()."""
    model = RecordingModel(["pred_a", "pred_b"])
    holdout_report = pd.DataFrame(
        {"precision": [1.0], "recall": [0.5]}, index=["class_a"]
    )
    payload = build_payload(
        model,
        holdout_metrics={"accuracy": 0.8, "macro_f1": 0.7},
        holdout_report=holdout_report,
        cv_mean_metrics={"accuracy": 0.6},
        final_metrics={"accuracy": 0.4},
    )
    classifier, pipeline = make_classifier(
        train_data=["a", "b"],
        labels=["x", "y"],
        payload=payload,
        discovered_features=["alpha", "beta"],
    )

    classifier.fit()

    assert pipeline._discover_calls == 1
    assert pipeline._execute_calls == 1
    assert pipeline._last_active_features == ["alpha", "beta"]
    assert classifier.model is model
    assert classifier.metrics == {"accuracy": 0.8, "macro_f1": 0.7}
    pd.testing.assert_frame_equal(classifier.report, holdout_report)
    assert classifier._results_payload["final_model"] is model
    assert classifier._results_payload["holdout_metrics"] == {
        "accuracy": 0.8,
        "macro_f1": 0.7,
    }
    pd.testing.assert_frame_equal(
        classifier._results_payload["holdout_report"], holdout_report
    )


def test_fit_calls_initialize_model_hook_before_training():
    """Tests fit() initializes a model before executing training."""
    model = RecordingModel(["pred_a", "pred_b"])
    classifier = TrackingClassifier(
        train_data=["a", "b"],
        labels=["x", "y"],
        pipeline=DummyPipeline(
            discovered_features=["alpha", "beta"],
            training_payload=build_payload(model),
        ),
    )

    classifier.fit()

    assert classifier._initialize_calls == 1


def test_predict_before_fit_raises_value_error():
    """Tests Value error is raised if predict() is performed before fit()."""
    classifier, _ = make_classifier(
        train_data=["a", "b"],
        labels=["x", "y"],
        payload=build_payload(RecordingModel(["pred_a", "pred_b"])),
    )

    with pytest.raises(ValueError):
        classifier.predict(np.array([[0.0], [1.0]]))


@pytest.mark.parametrize(
    ("keyword", "kwargs"),
    [
        ("ids", {"ids": ["only_one"]}),
        ("true_labels", {"true_labels": ["only_one"]}),
    ],
)
def test_predict_rejects_mismatched_metadata_lengths(keyword, kwargs):
    """Test ValueError is raised for predict() when ids or true_labels length does not match sample count."""
    model = RecordingModel(["pred_a", "pred_b"])
    classifier, _ = make_classifier(
        train_data=["a", "b"],
        labels=["x", "y"],
        payload=build_payload(model),
    )
    classifier.fit()

    with pytest.raises(ValueError):
        classifier.predict(np.array([[0.0], [1.0]]), **kwargs)


@pytest.mark.parametrize("scaler", [None, OffsetScaler(offset=1.0)])
def test_predict_returns_dataframe_and_handles_scaler(scaler):
    """Verify predict returns a formatted DataFrame and correctly applies feature scaling when present."""
    model = RecordingModel(["pred_a", "pred_b"])
    classifier, _ = make_classifier(
        train_data=["a", "b"],
        labels=["x", "y"],
        payload=build_payload(model, scaler=scaler),
    )
    classifier.fit()

    data = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = classifier.predict(
        data,
        ids=["doc-1", "doc-2"],
        true_labels=["actual-1", "actual-2"],
    )

    assert list(result.columns) == ["doc_id", "true_label", "predicted_label"]
    assert result["doc_id"].tolist() == ["doc-1", "doc-2"]
    assert result["true_label"].tolist() == ["actual-1", "actual-2"]
    assert result["predicted_label"].tolist() == ["pred_a", "pred_b"]

    expected_input = data if scaler is None else data + 1.0
    np.testing.assert_allclose(
        np.asarray(model.last_input, dtype=float), expected_input
    )
    if scaler is None:
        assert model.last_input is data
    else:
        assert scaler.transform_calls == 1
        np.testing.assert_allclose(np.asarray(scaler.last_input, dtype=float), data)


def test_predict_filters_dataframe_inputs_through_pipeline_hook():
    """Tests predict() filters DataFrame inputs using the pipeline feature filtering hook."""
    model = RecordingModel(["pred_a", "pred_b"])
    pipeline = FilteringPipeline(
        discovered_features=["alpha", "beta"],
        training_payload=build_payload(model),
    )
    classifier = Classifier(
        train_data=["a", "b"],
        labels=["x", "y"],
        pipeline=pipeline,
        features=["alpha", "beta"],
    )
    classifier.fit()

    data = pd.DataFrame({"alpha": [1.0, 2.0], "beta": [3.0, 4.0], "gamma": [5.0, 6.0]})

    result = classifier.predict(data)

    assert pipeline._filter_calls == 1
    assert pipeline._last_filter_args == {
        "baseline_features": ["alpha", "beta"],
        "active_features": ["alpha", "beta"],
        "matrix_shape": (2, 3),
    }
    np.testing.assert_allclose(
        np.asarray(model.last_input, dtype=float),
        np.asarray([[1.0, 3.0], [2.0, 4.0]]),
    )
    assert result["predicted_label"].tolist() == ["pred_a", "pred_b"]


def test_predict_tokenizes_raw_text_inputs_before_vectorizing(monkeypatch):
    """Tests predict() tokenizes raw text inputs using the pipeline tokenizer before vectorization."""
    model = RecordingModel(["pred_a", "pred_b"])
    dtm = DummyDTM()
    classifier, _ = make_classifier(
        train_data=["a", "b"],
        labels=["x", "y"],
        payload=build_payload(model, final_dtm=dtm),
        discovered_features=["alpha", "beta"],
        include_bigrams=True,
    )
    classifier.fit()

    def fake_tokenize_items(items, include_bigrams=True):
        token_lists = []
        for item in items:
            tokens = [token.lower() for token in item.split() if token.isalpha()]
            if include_bigrams:
                tokens = (
                    tokens + [f"{tokens[0]}_{tokens[1]}"]
                    if len(tokens) >= 2
                    else tokens
                )
            token_lists.append(tokens)
        return token_lists

    fake_lexos = types.ModuleType("lexos")
    fake_lexos.__path__ = []
    fake_classification = types.ModuleType("lexos.classification")
    fake_classification.__path__ = []
    fake_mlp_pipeline = types.ModuleType("lexos.classification.mlp_pipeline")
    fake_mlp_pipeline._tokenize_items = fake_tokenize_items

    monkeypatch.setitem(sys.modules, "lexos", fake_lexos)
    monkeypatch.setitem(sys.modules, "lexos.classification", fake_classification)
    monkeypatch.setitem(
        sys.modules, "lexos.classification.mlp_pipeline", fake_mlp_pipeline
    )

    result = classifier.predict(["Alpha beta", "Gamma delta"])

    assert dtm.vectorizer.last_input == [
        ["alpha", "beta", "alpha_beta"],
        ["gamma", "delta", "gamma_delta"],
    ]
    np.testing.assert_allclose(
        np.asarray(model.last_input, dtype=float),
        np.asarray([[0.0, 0.5], [1.0, 1.5]]),
    )
    assert result["predicted_label"].tolist() == ["pred_a", "pred_b"]


def test_split_preserves_label_distribution_and_proportions():
    """Tests split() is appropriately splitting train and test sets."""
    train_data = [f"doc_{index}" for index in range(40)]
    labels = ["class_a"] * 20 + ["class_b"] * 20
    classifier, _ = make_classifier(
        train_data=train_data,
        labels=labels,
        payload=build_payload(RecordingModel(["pred_a", "pred_b"])),
    )

    train_x, test_x, train_y, test_y = classifier.split(test_size=0.25, random_state=7)

    assert len(train_x) == 30
    assert len(test_x) == 10
    assert Counter(train_y) == Counter({"class_a": 15, "class_b": 15})
    assert Counter(test_y) == Counter({"class_a": 5, "class_b": 5})


# TODO: update this test when new feature removal algorithms are added
@pytest.mark.parametrize("feature_removal,seed", [(None, 11), ("random", 11)])
def test_feature_importance_sweep_prunes_features_in_expected_order(
    feature_removal, seed
):
    """Tests features are removed in correct order in feature_importance_sweep()."""
    features = ["alpha", "beta", "gamma"]
    expected_order = list(features)
    if feature_removal == "random":
        random.Random(seed).shuffle(expected_order)

    model = RecordingModel(["baseline_pred", "step_1_pred", "step_2_pred"])
    payload = build_payload(
        model,
        holdout_metrics={"accuracy": 0.75, "macro_f1": 0.7},
        cv_mean_metrics={"accuracy": 0.72, "macro_f1": 0.68},
        final_metrics={"accuracy": 0.8, "macro_f1": 0.77},
    )
    classifier, _ = make_classifier(
        train_data=["a", "b", "c"],
        labels=["x", "y", "x"],
        payload=payload,
        discovered_features=features,
        features=features,
        feature_removal=feature_removal,
        seed=seed,
    )

    result = classifier.feature_importance_sweep()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(features) + 1
    assert result["configuration"].tolist() == [
        "baseline",
        "remove_01",
        "remove_02",
        "remove_03",
    ]
    assert result["removed_feature"].tolist() == ["baseline", *expected_order]
    assert result["features_remaining"].tolist() == [3, 2, 1, 0]
    assert {"holdout_accuracy", "cv_accuracy", "final_model_accuracy"}.issubset(
        result.columns
    )


def test_feature_importance_sweep_discovers_features_when_missing():
    """Tests feature_importance_sweep triggers feature discovery when features are not explicitly set."""
    model = RecordingModel(["baseline_pred", "step_1_pred", "step_2_pred"])
    payload = build_payload(
        model,
        holdout_metrics={"accuracy": 0.75, "macro_f1": 0.7},
        cv_mean_metrics={"accuracy": 0.72, "macro_f1": 0.68},
        final_metrics={"accuracy": 0.8, "macro_f1": 0.77},
    )
    classifier, pipeline = make_classifier(
        train_data=["a", "b", "c"],
        labels=["x", "y", "x"],
        payload=payload,
        discovered_features=["alpha", "beta"],
        features=None,
    )

    result = classifier.feature_importance_sweep()

    assert pipeline._discover_calls == 1
    assert classifier.features == ["alpha", "beta"]
    assert result["configuration"].tolist() == ["baseline", "remove_01", "remove_02"]
    assert result["removed_feature"].tolist() == ["baseline", "alpha", "beta"]
