"""trainer.py.

An object-oriented classification core framework for Lexos.
Utilizes the Strategy and Template Method design patterns to allow for a hybrid,
input-agnostic context orchestration loop where individual Pipelines dictate
the underlying data transformation strategies.

Last Updated: August 18, 2026
"""

import copy
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator


class Pipeline(BaseModel, ABC):
    """Abstract base class representing a model training strategy.

    Subclasses dictate how raw dataset sequences (DataFrames, text streams, or arrays)
    are transformed, normalized, and trained
    """

    min_df: int = Field(default=2, description="Minimum data document expression limit")
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def execute_training(
        self, train_data: Sequence[Any] | pd.DataFrame, labels: Sequence[str]
    ) -> dict[str, Any]:
        """Execute feature extraction and model fitting routines.

        Args:
            train_data: Input data sequences (could be a pd.DataFrame, strings, or pre-vectorized arrays).
            labels: Targer categories corresponding directly to rain_data.
            active_features: Explicit subset of feature names/columns to preserve during training sweeps.

        Returns:
            A dictionary containing structural evaluations, metrics, and models.
        """
        pass

    @abstractmethod
    def discover_features(self, train_data: Sequence[Any] | pd.DataFrame) -> list[str]:
        """Discovers and returns available baseline feature/column names from the input data layout.

        This delegates feature space tracking to the individual Pipeline strategies
        """
        pass


class Classifier(BaseModel):
    """The central orchestration Context utilizing the Template Method pattern.

    Defines the structural lifecycle workflow for training classification tasks, while being agnostic to
    the input data type formats, delagting feature manipulation to the Pipeline strategies.
    """

    train_data: Any = Field(
        description="Training data source (e.g., list of strings, pre-computed DtaFrames, or arrays)"
    )
    labels: Sequence[str] = Field(description="Classification target identifiers")
    pipeline: Pipeline = Field(description="Injected configuration training strategy")
    features: Optional[Any] = Field(
        default=None,
        description="Explicit list of active features keys/columns to track, or None for auto-discovery",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Private fields mapping downstream evaluation states safely
    _metrics: dict[str, float] = PrivateAttr(default_factory=dict)
    _report: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)
    _model: Optional[Any] = PrivateAttr(default=None)

    _results_payload: dict[str, Any] = PrivateAttr(default_factory=dict)

    @field_validator("train_data")
    @classmethod
    def _validate_train_data(cls, value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return value

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value

        raise TypeError(
            "train_data must be a pandas DataFrame or a sequence of training samples."
        )

    @property
    def metrics(self) -> dict[str, float]:
        """Expose calculated model validation performance metrics."""
        return self._metrics

    @property
    def report(self) -> pd.DataFrame:
        """Expose structured classification precision/recall dataframe reports."""
        return self._report

    @property
    def model(self) -> Any:
        """Expose the underlying trained estimator payload."""
        return self._model

    def predict(
        self,
        data: Any,
        ids: Optional[Sequence[Any]] = None,
        true_labels: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Predict labels for a fitted classifier using the stored final scaler and model.

        Args:
            data: Input feature data (e.g., pd.DataFrame or pre-vectorized array).
            ids: Optional sequence of document or sample identifiers.
            true_labels: Optional sequence of the actual ground truth labels.

        Returns:
            pd.DataFrame: A structured table combining IDs, True Labels (if provided),
                          and the Model's Predicted Labels.
        """
        if self._model is None:
            raise ValueError("Classifier must be fitted before calling predict().")

        scaler = self._results_payload.get("final_scaler")

        # 1. Safely handle feature alignment if data is a DataFrame
        if isinstance(data, pd.DataFrame):
            # Ensure columns match what the pipeline expects, using the baseline features
            data = self.pipeline._filter_active_features(
                baseline_features=self.features,
                matrix=data,
                active_features=self.features,
            )

        elif isinstance(data, list) and isinstance(data[0], str):
            from lexos.classification.utils import _tokenize_items

            dtm_object = self._results_payload.get("final_dtm")

            data = dtm_object.vectorizer.transform(
                [
                    _tokenize_items(
                        [text], include_bigrams=self.pipeline.include_bigrams
                    )[0]
                    for text in data
                ]
            )

        # 2. Scale and run inference
        transformed_data = scaler.transform(data) if scaler is not None else data
        predictions = self._model.predict(transformed_data)

        # 3. Assemble the rich output DataFrame
        output_dict = {}

        # Add Document IDs if given, otherwise fallback to standard integer indices
        if ids is not None:
            if len(ids) != len(predictions):
                raise ValueError("Length of 'ids' must match length of predictions.")
            output_dict["doc_id"] = list(ids)
        else:
            output_dict["doc_id"] = list(range(len(predictions)))

        # Add True Labels if provided for side-by-side comparison
        if true_labels is not None:
            if len(true_labels) != len(predictions):
                raise ValueError(
                    "Length of 'true_labels' must match length of predictions."
                )
            output_dict["true_label"] = list(true_labels)

        # Add the final predictions
        output_dict["predicted_label"] = list(predictions)

        return pd.DataFrame(output_dict)

    def fit(self) -> None:
        """The Template Method establishing the explicit lifecycle algorithm sequence."""
        self._preprocess_data()
        self._initialize_model()
        results = self._train()
        self._evaluate(results)

    def _preprocess_data(self) -> None:
        """Hook reserved for shared dataset formatting and dynamic feature discovery."""
        if len(self.train_data) != len(self.labels):
            raise ValueError(
                "Size mismatch across sample dimensions and structural target labels."
            )

        # Delegate feature discovery to the Pipeline strategy
        if self.features is None:
            self.features = self.pipeline.discover_features(self.train_data)

    def _initialize_model(self) -> None:
        """Hook executed right before executing strategy processing layers."""
        pass

    def _train(self) -> dict[str, Any]:
        """Delegates feature production and model training to the injected Strategy."""
        return self.pipeline.execute_training(
            self.train_data, self.labels, active_features=self.features
        )

    def _evaluate(self, results: dict[str, Any]) -> None:
        """Populates internal context performance metrics from strategy payloads."""
        self._model = results.get("final_model")

        self._results_payload = results
        self._metrics = (
            results.get("holdout_metrics", {})
            or results.get("cv_mean_metrics", {})
            or results.get("final_metrics", {})
        )
        self._report = (
            results.get("holdout_report", pd.DataFrame())
            if not results.get("holdout_report", pd.DataFrame()).empty
            else results.get("final_report", pd.DataFrame())
        )

    def feature_importance_sweep(self) -> pd.DataFrame:
        """Natively retrains models by progressively pruning features based on the pipeline configuration.

        Handles cloning configurations and iterative drops automatically to insulate client environments.

        Returns:
            pd.DataFrame: A comprehensive table tracking performance metrics per feature removal step.
        """
        if self.features is None:
            self.features = self.pipeline.discover_features(self.train_data)

        active_features = list(self.features)
        removal_order = list(self.features)

        # Still lacking different removal strategies
        strategy_removal = getattr(self.pipeline, "feature_removal", None)
        if strategy_removal == "random":
            seed = getattr(self.pipeline, "seed", 42)
            random.Random(seed).shuffle(removal_order)

        experiment_rows = []

        self.fit()

        base_holdout = (
            getattr(self, "_results_payload", {}).get("holdout_metrics") or {}
        )
        base_cv = getattr(self, "_results_payload", {}).get("cv_mean_metrics") or {}
        base_final = getattr(self, "_results_payload", {}).get("final_metrics") or {}
        base_row = {
            "configuration": "baseline",
            "removed_feature": "baseline",
            "features_remaining": len(active_features),
            "holdout_accuracy": base_holdout.get("accuracy", np.nan),
            "holdout_balanced_accuracy": base_holdout.get("balanced_accuracy", np.nan),
            "holdout_macro_f1": base_holdout.get("macro_f1", np.nan),
            "cv_accuracy": base_cv.get("accuracy", np.nan),
            "cv_balanced_accuracy": base_cv.get("balanced_accuracy", np.nan),
            "cv_macro_f1": base_cv.get("macro_f1", np.nan),
            "final_model_accuracy": base_final.get("accuracy", np.nan),
            "final_model_macro_f1": base_final.get("macro_f1", np.nan),
        }
        experiment_rows.append(base_row)

        for step, feature_to_drop in enumerate(removal_order, start=1):
            active_features = [f for f in active_features if f != feature_to_drop]
            cloned_strategy = copy.deepcopy(self.pipeline)

            sub_classifier = Classifier(
                train_data=self.train_data,
                labels=self.labels,
                pipeline=cloned_strategy,
                features=active_features,
            )
            sub_classifier.fit()

            sub_holdout = (
                getattr(sub_classifier, "_results_payload", {}).get("holdout_metrics")
                or {}
            )
            sub_cv = (
                getattr(sub_classifier, "_results_payload", {}).get("cv_mean_metrics")
                or {}
            )
            sub_final = (
                getattr(sub_classifier, "_results_payload", {}).get("final_metrics")
                or {}
            )

            row = {
                "configuration": f"remove_{step:02d}",
                "removed_feature": feature_to_drop,
                "features_remaining": len(active_features),
                "holdout_accuracy": sub_holdout.get("accuracy", np.nan),
                "holdout_balanced_accuracy": sub_holdout.get(
                    "balanced_accuracy", np.nan
                ),
                "holdout_macro_f1": sub_holdout.get("macro_f1", np.nan),
                "cv_accuracy": sub_cv.get("accuracy", np.nan),
                "cv_balanced_accuracy": sub_cv.get("balanced_accuracy", np.nan),
                "cv_macro_f1": sub_cv.get("macro_f1", np.nan),
                "final_model_accuracy": sub_final.get("accuracy", np.nan),
                "final_model_macro_f1": sub_final.get("macro_f1", np.nan),
            }
            experiment_rows.append(row)

        return pd.DataFrame(experiment_rows)

    def split(
        self, test_size: float = 0.2, random_state: Optional[int] = None
    ) -> tuple[list[Any], list[Any], list[str], list[str]]:
        """Splits datasets safely before initializing pipeline contexts to avoid leakage."""
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(self.train_data))
        tr_idx, ts_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels,
        )

        train_x = [self.train_data[i] for i in tr_idx]
        test_x = [self.train_data[i] for i in ts_idx]
        train_y = [self.labels[i] for i in tr_idx]
        test_y = [self.labels[i] for i in ts_idx]

        return train_x, test_x, train_y, test_y
