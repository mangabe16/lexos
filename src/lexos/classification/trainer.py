"""trainer.py.

An object-oriented classification core framework for Lexos.
Utilizes the Strategy and Template Method design patterns to allow
modular training configurations and unified lifecycle evaluation.

Last Updated: June 22, 2026
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

from lexos.corpus.corpus_stats import CorpusStats


class Pipeline(BaseModel, ABC):
    """Abstract base class representing a model training strategy.

    Subclasses must implement specific feature extraction, tokenization,
    and model fitting execution loops.
    """

    min_df: int = Field(default=2, description="Minimum data document expression limit")
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def execute_training(
        self, train_data: Sequence[Any], labels: Sequence[str]
    ) -> dict[str, Any]:
        """Execute feature extraction and model fitting routines.

        Args:
            train_data: A sequence of text or preprocessed spaCy Doc objects.
            labels: Target categories corresponding directly to train_data.

        Returns:
            A dictionary containing structural evaluations, metrics, and models.
        """
        pass


class SklearnClassifierPipeline(Pipeline):
    """A concrete pipeline strategy handling traditional scikit-learn estimators.

    Ported from the legacy functional implementation to adapt smoothly to
    the new object-oriented framework.
    """

    model_name: str = Field(
        default="svc", description="The lowercase scikit-learn key name"
    )
    normalize: Optional[str] = Field(
        default=None, description="Normalization technique ('standard', 'minmax', etc.)"
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary estimator hyperparameters"
    )

    def _get_scaler(self) -> Any:
        """Instantiate the requested feature normalization model."""
        mapping = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "l2": lambda: Normalizer(norm="l2"),
        }
        if self.normalize not in mapping:
            raise ValueError(f"Unsupported normalization method: {self.normalize}")
        return mapping[self.normalize]()

    def _get_estimator(self) -> Any:
        """Instantiate the raw estimator underlying this execution profile."""
        registry = {
            "svc": lambda: SVC(kernel="linear", **self.model_kwargs),
            "logistic": lambda: LogisticRegression(max_iter=1000, **self.model_kwargs),
            "decision_tree": lambda: DecisionTreeClassifier(**self.model_kwargs),
            "random_forest": lambda: RandomForestClassifier(**self.model_kwargs),
            "knn": lambda: KNeighborsClassifier(**self.model_kwargs),
            "naive_bayes": lambda: MultinomialNB(**self.model_kwargs),
        }
        key = self.model_name.lower()
        if key not in registry:
            raise ValueError(f"Unknown model architecture variant: '{self.model_name}'")
        return registry[key]()

    def execute_training(
        self, train_data: Sequence[Any], labels: Sequence[str]
    ) -> dict[str, Any]:
        """Transforms features, normalizes distributions, and fits the estimator."""
        features = train_data
        scaler = None

        if self.normalize:
            scaler = self._get_scaler()
            features = scaler.fit_transform(features)

        estimator = self._get_estimator()
        estimator.fit(features, labels)

        return {
            "final_model": estimator,
            "final_scaler": scaler,
            "holdout_metrics": {},
            "holdout_report": pd.DataFrame(),
            "holdout_confusion_matrix": pd.DataFrame(),
            "cv_fold_metrics": pd.DataFrame(),
            "cv_mean_metrics": {},
        }


class Classifier(BaseModel):
    """The central orchestration Context utilizing the Template Method pattern.

    Defines the structural lifecycle workflow for training classification tasks.
    """

    train_data: Sequence[Any] = Field(
        description="Training data source (e.g., list of strings or spaCy Docs)"
    )
    labels: Sequence[str] = Field(description="Classification target identifiers")
    pipeline: Pipeline = Field(description="Injected configuration training strategy")
    features: Optional[Any] = Field(
        default="all", description="Features selector context, specific list, or 'all' for dynamic extraction"
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

    def build_corpus_stat_features(self) -> list[str]:
        """Dynamically discovers available numeric CorpusStats features for the current training dataset.

        Accessible both as an internal lifecycle step and as a public API tool for end-users.

        Returns:
            A list of string feature names discovered within the document matrix.
        """
        # Minor local imports to ensure clean initialization boundaries
        from lexos.classification.mlp_pipeline import _tokenize_items

        token_lists = _tokenize_items(self.train_data, include_bigrams=True)
        doc_ids = [f"doc_{i}" for i in range(len(token_lists))]
        
        docs_for_corpus = [
            (doc_id, doc_id, " ".join(tokens))
            for doc_id, tokens in zip(doc_ids, token_lists)
        ]
        
        corpus = CorpusStats(docs=docs_for_corpus, min_df=self.pipeline.min_df)
        stats_df = corpus.doc_stats_df.reindex(doc_ids)
        numeric_stats = stats_df.select_dtypes(include=[np.number]).copy()

        if numeric_stats.empty:
            raise ValueError("CorpusStats did not produce any numeric features to evaluate.")

        return list(numeric_stats.columns)

    def fit(self) -> None:
        """The Template Method establishing the explicit lifecycle algorithm sequence."""
        self._preprocess_data()
        self._initialize_model()
        results = self._train()
        self._evaluate(results)

    def _preprocess_data(self) -> None:
        """Hook reserved for shared dataset formatting and dynamic feature discovery validation."""
        if len(self.train_data) != len(self.labels):
            raise ValueError(
                "Size mismatch across sample dimensions and structural target labels."
            )

        # Dynamic discovery execution hook
        if self.features == "all":
            self.features = self.build_corpus_stat_features()

    def _initialize_model(self) -> None:
        """Hook executed right before executing strategy processing layers."""
        pass

    def _train(self) -> dict[str, Any]:
        """Delegates feature production and model training to the injected Strategy."""
        return self.pipeline.execute_training(self.train_data, self.labels)

    def _evaluate(self, results: dict[str, Any]) -> None:
        """Populates internal context performance metrics from strategy payloads."""
        self._model = results.get("final_model")
        self._metrics = results.get("holdout_metrics", {}) or results.get(
            "cv_mean_metrics", {}
        )
        self._report = results.get("holdout_report", pd.DataFrame())

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