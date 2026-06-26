"""trainer.py.

An object-oriented classification core framework for Lexos.
Utilizes the Strategy and Template Method design patterns to allow for a hybrid,
input-agnostic context orchestration loop where individual Pipelines dictate
the underlying data transformation strategies.

Last Updated: June 23, 2026
"""

import copy
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class Pipeline(BaseModel, ABC):
    """Abstract base class representing a model training strategy.

    Subclasses dictate how raw dataset sequences (DataFrames, text streams, or arrays)
    are transformed, normalized, and trained
    """

    min_df: int = Field(default=2, description="Minimum data document expression limit")
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def execute_training(
        self, train_data: Sequence[Any], labels: Sequence[str]
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
    def discover_features(self, train_data: Sequence[Any]) -> list[str]:
        """Discovers and returns available baseline feature/column names from the input data layout.

        This delegates feature space tracking to the individual Pipeline strategies
        """
        pass


class Classifier(BaseModel):
    """The central orchestration Context utilizing the Template Method pattern.

    Defines the structural lifecycle workflow for training classification tasks, while being agnostic to
    the input data type formats, delagting feature manipulation to the Pipeline strategies.
    """

    train_data: Sequence[Any] = Field(
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

    def fit(self) -> None:
        """The Template Method establishing the explicit lifecycle algorithm sequence."""
        self._preprocess_data()
        self._initialize_model()
        results = self._train()
        self._evaluate(results)

    def _preprocess_data(self) -> None:
        """Hook reserved for shared dataset formatting and dynamic feature discovery discovery."""
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
        self._metrics = results.get("holdout_metrics", {}) or results.get(
            "cv_mean_metrics", {}
        )
        self._report = results.get("holdout_report", pd.DataFrame())

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

        strategy_removal = getattr(self.pipeline, "feature_removal", None)
        # TODO: add more removal strategies
        if strategy_removal == "random":
            seed = getattr(self.pipeline, "seed", 42)
            random.Random(seed).shuffle(removal_order)

        experiment_rows = []

        # 1. Train and parse baseline configurations
        self.fit()
        base_row = {
            "configuration": "baseline",
            "removed_feature": "baseline",
            "features_remaining": len(active_features),
            "holdout_accuracy": self.metrics.get("accuracy", 0.0),
            "holdout_balanced_accuracy": self.metrics.get("balanced_accuracy", 0.0),
            "holdout_macro_f1": self.metrics.get("macro_f1", 0.0),
            "cv_accuracy": self.metrics.get("accuracy", 0.0),
            "cv_balanced_accuracy": self.metrics.get("balanced_accuracy", 0.0),
            "cv_macro_f1": self.metrics.get("macro_f1", 0.0),
        }
        experiment_rows.append(base_row)

        # 2. Progressively drop features one by one
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

            sub_metrics = sub_classifier.metrics
            row = {
                "configuration": f"remove_{step:02d}",
                "removed_feature": feature_to_drop,
                "features_remaining": len(active_features),
                "holdout_accuracy": sub_metrics.get("accuracy", 0.0),
                "holdout_balanced_accuracy": sub_metrics.get("balanced_accuracy", 0.0),
                "holdout_macro_f1": sub_metrics.get("macro_f1", 0.0),
                "cv_accuracy": sub_metrics.get("accuracy", 0.0),
                "cv_balanced_accuracy": sub_metrics.get("balanced_accuracy", 0.0),
                "cv_macro_f1": sub_metrics.get("macro_f1", 0.0),
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
