"""test_decision_tree_pipeline.py.

Coverage: 100%
Last update: August 25, 2026
Last test: August 25, 2026

Unit tests for the DecisionTreePipeline class.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from lexos.classification.decision_tree_pipeline import DecisionTreePipeline


def _training_corpus() -> tuple[list[str], list[str]]:
    """Return a small corpus with distinct vocabulary for two classes."""
    documents = [
        "commerce treasury federal",
        "treasury federal debt",
        "liberty republic rights",
        "republic rights people",
    ]
    labels = ["HAMILTON", "HAMILTON", "MADISON", "MADISON"]
    return documents, labels


def test_discover_features_respects_bigram_setting():
    """Discover unigrams only when bigram generation is disabled."""
    documents, _ = _training_corpus()

    unigrams = DecisionTreePipeline(min_df=1, include_bigrams=False).discover_features(
        documents
    )
    with_bigrams = DecisionTreePipeline(
        min_df=1, include_bigrams=True
    ).discover_features(documents)

    assert {"commerce", "treasury", "federal", "liberty", "republic", "rights"} <= set(
        unigrams
    )
    assert all("_" not in feature for feature in unigrams)
    assert "commerce_treasury" in with_bigrams
    assert set(unigrams) < set(with_bigrams)


def test_discover_features_accepts_token_lists():
    """Discover features from already-tokenized documents."""
    documents = [["alpha", "beta"], ["beta", "gamma"]]

    features = DecisionTreePipeline(min_df=1, include_bigrams=False).discover_features(
        documents
    )

    assert set(features) == {"alpha", "beta", "gamma"}


def test_execute_training_returns_fitted_decision_tree_payload():
    """Fit a tree and return the structural payload expected by Classifier."""
    documents, labels = _training_corpus()
    pipeline = DecisionTreePipeline(
        min_df=1,
        include_bigrams=False,
        seed=17,
        tree_kwargs={"max_depth": 2},
    )

    results = pipeline.execute_training(documents, labels)

    assert isinstance(results["final_model"], DecisionTreeClassifier)
    assert results["final_model"].random_state == 17
    assert results["final_model"].max_depth == 2
    assert results["final_model"].n_features_in_ == len(
        pipeline.discover_features(documents)
    )
    assert results["final_dtm"].shape == (
        len(documents),
        results["final_model"].n_features_in_,
    )
    assert results["final_scaler"] is None
    assert results["holdout_metrics"] == {}
    assert results["cv_mean_metrics"] == {}
    assert isinstance(results["holdout_report"], pd.DataFrame)
    assert isinstance(results["holdout_confusion_matrix"], pd.DataFrame)
    assert isinstance(results["cv_fold_metrics"], pd.DataFrame)


def test_execute_training_restricts_matrix_to_active_features():
    """Fit using only the requested features that exist in the discovered vocabulary."""
    documents, labels = _training_corpus()
    pipeline = DecisionTreePipeline(min_df=1, include_bigrams=False)
    active_features = ["treasury", "missing_feature"]

    results = pipeline.execute_training(
        documents, labels, active_features=active_features
    )

    estimator = results["final_model"]
    assert estimator.n_features_in_ == 1
    assert np.isfinite(estimator.tree_.threshold).all()


def test_predict_transforms_text_with_fitted_pipeline():
    """Predict text using the fitted DTM and estimator."""
    documents, labels = _training_corpus()
    pipeline = DecisionTreePipeline(min_df=1, include_bigrams=False, seed=17)
    results = pipeline.execute_training(documents, labels)

    predictions = pipeline.predict(["commerce treasury", "liberty republic"], results)

    assert len(predictions) == 2
    assert set(predictions) <= set(labels)


def test_predict_restricts_text_to_active_features():
    """Predict using the same selected feature subset used during training."""
    documents, labels = _training_corpus()
    pipeline = DecisionTreePipeline(min_df=1, include_bigrams=False, seed=17)
    active_features = ["treasury"]
    results = pipeline.execute_training(
        documents,
        labels,
        active_features=active_features,
    )

    predictions = pipeline.predict(
        ["treasury", "liberty"],
        results,
        active_features=active_features,
    )

    assert list(predictions) == ["HAMILTON", "MADISON"]


def test_predict_rejects_incomplete_payload():
    """Prediction requires both a fitted estimator and DTM."""
    pipeline = DecisionTreePipeline()

    with pytest.raises(ValueError, match="missing fitted model or DTM"):
        pipeline.predict(["text"], {})
