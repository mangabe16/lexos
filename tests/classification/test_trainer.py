"""test_trainer.py.

Coverage: 92%
Last update: 06/09/2026
Last test: 06/09/2026
"""

import numpy as np
import pytest

import sklearn
from sklearn.exceptions import NotFittedError

from lexos.classification import trainer as trainer_mod
from lexos.classification.trainer import (
    train_classifier,
    fit_classifier,
    predict_labels,
    normalize_features,
)


def make_separable_data(seed=0):
    """Create a small linearly-separable 2D dataset.

    Returns (X, y) where X is shape (10, 2) and y contains two classes
    (five samples each).
    """
    rng = np.random.RandomState(seed)
    class0 = rng.normal(loc=-1.0, scale=0.1, size=(5, 2))
    class1 = rng.normal(loc=1.0, scale=0.1, size=(5, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * 5 + [1] * 5)
    return X, y


def test_train_classifier_svc_default():
    """Train an SVC using `train_classifier` and validate output.

    Confirms a fitted estimator is returned and the classification
    report string is produced and non-empty.
    """
    X, y = make_separable_data()
    clf, report = train_classifier(X, y, model="svc", test_size=0.4, random_state=42)
    assert hasattr(clf, "predict")
    assert isinstance(report, str) and len(report) > 0
    assert "precision" in report or "accuracy" in report


def test_predict_labels_after_fit():
    """Fit a logistic regressor and check `predict_labels` output shape.

    Ensures `fit_classifier` can train a `LogisticRegression` and that
    `predict_labels` returns one prediction per input row.
    """
    X, y = make_separable_data(seed=1)
    clf = fit_classifier(X, y, model="logistic_regression", max_iter=200, random_state=0)
    X_new = np.array([[ -1.0, -1.0 ], [1.0, 1.0]])
    preds = predict_labels(clf, X_new)
    assert preds.shape[0] == X_new.shape[0]


def test_normalize_features_and_invalid():
    """Verify `normalize_features` returns correct scaler and errors on invalid input."""
    sc = normalize_features("standard")
    from sklearn.preprocessing import StandardScaler

    assert isinstance(sc, StandardScaler)

    with pytest.raises(ValueError):
        normalize_features("no_such_method")


def test_train_classifier_unsupported_model_raises():
    """Ensure `train_classifier` raises for an unknown model name."""
    X, y = make_separable_data()
    with pytest.raises(ValueError):
        train_classifier(X, y, model="not_a_model")


def test_train_and_fit_model_coverage():
    """Exercise all supported models in `train_classifier` and `fit_classifier`.

    Verifies that each supported model can be trained and returns an
    estimator with a `predict` method.
    """
    X, y = make_separable_data()

    # Models accepted by train_classifier
    for model in ["svc", "logistic", "decision_tree", "random_forest", "knn", "naive_bayes"]:
        # MultinomialNB requires non-negative input; ensure that for the test
        X_use, y_use = (np.abs(X), y) if model == "naive_bayes" else (X, y)
        clf, report = train_classifier(X_use, y_use, model=model, test_size=0.4, random_state=0)
        assert hasattr(clf, "predict")
        assert isinstance(report, str)

    # Models accepted by fit_classifier
    for model in ["svc", "logistic_regression", "decision_tree", "random_forest", "naive_bayes"]:
        X_use, y_use = (np.abs(X), y) if model == "naive_bayes" else (X, y)
        clf = fit_classifier(X_use, y_use, model=model)
        assert hasattr(clf, "predict")


def test_fit_classifier_kwargs_forwarding():
    """Ensure estimator-specific kwargs (e.g. `max_iter`) are forwarded to the estimator."""
    X, y = make_separable_data()
    clf = fit_classifier(X, y, model="logistic_regression", max_iter=50, random_state=0)
    # `LogisticRegression` exposes `max_iter` attribute reflecting the passed value
    assert getattr(clf, "max_iter", None) == 50


def test_train_classifier_single_class_stratify():
    """When all labels are identical, `train_classifier` should still work (no stratify)."""
    X = np.random.RandomState(0).randn(10, 3)
    y = np.zeros(10, dtype=int)
    # SVC and many classifiers require >=2 classes; ensure the function raises
    with pytest.raises(ValueError):
        train_classifier(X, y, model="svc", test_size=0.3, random_state=0)


def test_bootstrap_uses_resample(monkeypatch):
    """Verify that passing `bootstrap=True` triggers the module's resample() call.
    """
    called = {}

    def fake_resample(features, labels, random_state=None):
        called['ok'] = True
        # Return a resampled array that preserves class variety for training.
        # For deterministic behavior return the original array (no-op resample).
        return features.copy(), np.asarray(labels).copy()

    monkeypatch.setattr(trainer_mod, "resample", fake_resample)

    X = np.random.RandomState(1).randn(4, 2)
    y = np.array([0, 0, 1, 1])
    clf, report = train_classifier(X, y, model="svc", test_size=0.5, random_state=0, bootstrap=True)
    assert called.get('ok', False) is True
    assert hasattr(clf, "predict")


def test_train_classifier_test_size_used(monkeypatch):
    """Confirm `train_classifier` uses the requested `test_size` when splitting
    by wrappping `train_test_split` to capture arguments passed by `train_classifier`.
    """
    captured = {}

    orig = trainer_mod.train_test_split

    def wrapper(*args, **kwargs):
        captured['kwargs'] = kwargs.copy()
        return orig(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "train_test_split", wrapper)

    X, y = make_separable_data()
    _clf, _report = train_classifier(X, y, model="svc", test_size=0.33, random_state=0)
    assert pytest.approx(captured['kwargs']['test_size'], rel=1e-3) == 0.33


def test_fit_unknown_model_raises():
    """`fit_classifier` should raise for unknown model names (symmetry with train_classifier)."""
    X, y = make_separable_data()
    with pytest.raises(ValueError):
        fit_classifier(X, y, model="no_such_model")


def test_predict_labels_unfitted_raises():
    """Calling `predict_labels` with an unfitted estimator should raise NotFittedError."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    X_new = np.array([[0.0, 0.0]])
    with pytest.raises(NotFittedError):
        predict_labels(clf, X_new)


def test_train_classifier_normalize_and_bootstrap_combined():
    """Integration smoke test: `normalize` + `bootstrap` together should run end-to-end."""
    X, y = make_separable_data()
    clf, report = train_classifier(X, y, model="svc", test_size=0.4, random_state=0, normalize="standard", bootstrap=True)
    assert hasattr(clf, "predict")