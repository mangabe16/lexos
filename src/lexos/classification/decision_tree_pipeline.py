"""decision_tree_pipeline.py.

A minimal, text-functional Decision Tree Pipeline Strategy for Lexos.
Handles its own tokenization and DTM generation internally.

Last Updated: June 26, 2026
"""

from typing import Any, Sequence, Optional
import pandas as pd
from pydantic import Field
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import scipy.sparse as sp

from lexos.dtm import DTM
from lexos.classification.trainer import Pipeline
from lexos.classification.mlp_pipeline import _tokenize_items, _to_dense


class DecisionTreePipeline(Pipeline):
    """A minimal text pipeline strategy executing Decision Tree classification."""

    seed: int = Field(default=42, description="Random state initialization seed")
    include_bigrams: bool = Field(
        default=True, description="Enables bigram token profiles"
    )
    tree_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Estimator hyperparameters"
    )

    def execute_training(
        self,
        train_data: Sequence[Any],
        labels: Sequence[str],
        active_features: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Fits the Decision Tree model on compiled numerical features."""
        baseline_features = self.discover_features(
            train_data
        )  # Full baseline string features list

        # Extracting the numerical matrix for training that matches the raw documents
        token_lists = _tokenize_items(train_data, include_bigrams=self.include_bigrams)
        doc_labels = [f"train_doc_{i}" for i in range(len(token_lists))]

        dtm = DTM()
        dtm(token_lists, labels=doc_labels, min_df=self.min_df)
        x_raw = dtm.doc_term_matrix

        if active_features is not None:
            features_indices = [
                baseline_features.index(feature)
                for feature in active_features
                if feature in baseline_features
            ]
            x_raw = (
                x_raw[:, features_indices]
                if isinstance(x_raw, (sp.spmatrix, np.ndarray))
                else dtm.tocsr()[:, features_indices]
            )

        # Model Initialization and Fitting
        estimator = DecisionTreeClassifier(random_state=self.seed, **self.tree_kwargs)
        estimator.fit(_to_dense(x_raw), labels)

        # Return the payload to the main Classifier engine
        return {
            "final_model": estimator,
            "final_dtm": dtm,
            "final_scaler": None,
            "holdout_metrics": {},
            "holdout_report": pd.DataFrame(),
            "holdout_confusion_matrix": pd.DataFrame(),
            "cv_fold_metrics": pd.DataFrame(),
            "cv_mean_metrics": {},
        }

    def discover_features(self, train_data: Sequence[Any]) -> list[str]:
        """Extracts text features via DTM."""
        token_lists = _tokenize_items(train_data, include_bigrams=self.include_bigrams)
        doc_labels = [f"train_doc_{i}" for i in range(len(token_lists))]

        # Instantiate empty DTM model object
        dtm = DTM()

        # Call instance to fit vectorizer and construct matrix
        dtm(token_lists, labels=doc_labels, min_df=self.min_df)

        # Access term list property on the dtm object directly
        return dtm.sorted_terms_list
