"""mlp_pipeline.py.

A leakage-safe Multi-Layer Perceptron Text Classification Pipeline Strategy
Last Updated: June 26, 2026
"""

from typing import Any, Sequence, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import Field
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

from lexos.dtm import DTM
from lexos.tokenizer import WhitespaceTokenizer
from lexos.tokenizer.ngrams import Ngrams
from lexos.classification.trainer import Pipeline


def _tokenize_items(
    items: Sequence[Any], include_bigrams: bool = True
) -> list[list[str]]:
    """Transform sequence objects or strings into normal unigram/bigram token vectors."""
    ws_tokenizer = WhitespaceTokenizer()
    ngrams = Ngrams(n=2)
    token_lists: list[list[str]] = []

    for item in items:
        if hasattr(item, "text"):
            unigrams = [tok.text.lower() for tok in item if tok.is_alpha]
        elif isinstance(item, str):
            unigrams = [tok for tok in ws_tokenizer(item.lower()) if tok.isalpha()]
        else:
            unigrams = [str(tok).lower() for tok in item if str(tok).isalpha()]

        if include_bigrams:
            bigrams = [
                f"{t1}_{t2}"
                for t1, t2 in ngrams.from_tokens(unigrams, output="tuples")
                if t1.isalpha() and t2.isalpha()
            ]
            tokens = unigrams + bigrams
        else:
            tokens = unigrams
        token_lists.append(tokens)

    if any(len(tokens) == 0 for tokens in token_lists):
        raise ValueError(
            "At least one document produced zero tokens after preprocessing."
        )
    return token_lists


def _to_dense(matrix: Any) -> np.ndarray:
    """Safely cast arbitrary matrices to a dense numpy array format."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


class MLPPipeline(Pipeline):
    """Configuration Strategy implementing custom Neural Network text classification flows."""

    seed: int = Field(default=42, description="Random state initialization seed")
    test_size: float = Field(
        default=0.2, description="Validation split layout size ratio"
    )
    cv_splits: int = Field(
        default=5, description="Cross-validation evaluation loop folding constraint"
    )
    include_bigrams: bool = Field(
        default=True, description="Enables bigram token compilation profiles"
    )
    use_smote: bool = Field(
        default=True,
        description="Controls implementation of SMOTE oversampling algorithms",
    )
    feature_removal: Optional[str] = Field(
        default=None,
        description="Set to 'sequential' or 'random' to trigger native importance pruning loops",
    )
    run_holdout: bool = Field(
        default=False,
        description="Controls execution of holdout model evaluation",
    )
    run_cv: bool = Field(
        default=False, description="Controls execution of cross-validation step"
    )
    mlp_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "hidden_layer_sizes": (64,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "max_iter": 1000,
        },
        description="Configuration keywords directly targeting scikit-learn MLP instances",
    )

    def discover_features(self, train_data: Sequence[Any]) -> list[str]:
        """Discovers and returns availble baseline features based on input data type."""
        # Handles case where user provides a dataframe of features
        if isinstance(train_data, pd.DataFrame):
            return train_data.columns.tolist()

        # Handles case where user provides the raw text
        if isinstance(train_data, list | str):
            features_DTM = DTM()
            tokenized_data = _tokenize_items(
                train_data, include_bigrams=self.include_bigrams
            )
            mock_labels = [f"doc_{i}" for i in range(len(tokenized_data))]
            features_DTM.fit_transform(
                tokenized_data, labels=mock_labels, min_df=self.min_df
            )
            return features_DTM.sorted_terms_list

        return []

    def _normalize_active_features(
        self, active_features: Optional[Any]
    ) -> Optional[list[str]]:
        """Normalize feature selectors to a plain list of column names."""
        if active_features is None:
            return None

        if isinstance(active_features, pd.DataFrame):
            return active_features.columns.tolist()

        if isinstance(active_features, pd.Index):
            return active_features.tolist()

        if isinstance(active_features, str):
            return [active_features]

        return list(active_features)

    def _filter_active_features(
        self,
        baseline_features: list[str],
        matrix: Any,
        active_features: Optional[Any],
    ) -> Any:
        """Slices a matrix to only retain terms explicitly provided in active_features."""
        active_features = self._normalize_active_features(active_features)
        if active_features is None:
            return matrix

        if isinstance(matrix, pd.DataFrame):
            # Strict validation for DataFrames remains unchanged
            feature_positions = {
                feature: index for index, feature in enumerate(baseline_features)
            }
            missing_features = [
                feature
                for feature in active_features
                if feature not in feature_positions
            ]
            if missing_features:
                raise ValueError(
                    "Requested active features are not present in the baseline feature set: "
                    f"{missing_features}"
                )
            return matrix.loc[:, active_features]

        # For numerical text matrices (numpy/scipy sparse), tolerate sub-fold vocabulary shrinkage
        feature_positions = {
            feature: index for index, feature in enumerate(baseline_features)
        }

        # Only extract indices for features that actually exist in this split's vocabulary matrix
        feature_indices = [
            feature_positions[feature]
            for feature in active_features
            if feature in feature_positions
        ]

        if isinstance(matrix, np.ndarray):
            return matrix[:, feature_indices]
        return matrix.tocsr()[:, feature_indices]

    def _apply_smote(
        self, feature_matrix: Any, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_smote:
            return feature_matrix, labels

        dense_matrix = _to_dense(feature_matrix)
        if SMOTE is None:
            raise ImportError("SMOTE requested, but imbalanced-learn is not installed.")

        smote = SMOTE(random_state=self.seed)
        return smote.fit_resample(dense_matrix, labels)

    def _build_mlp_classifier(self) -> MLPClassifier:
        """Instantiate an explicit MLP Classifier using current keyword parameters."""
        kwargs = dict(self.mlp_kwargs)
        kwargs.setdefault("random_state", self.seed)
        return MLPClassifier(**kwargs)

    def execute_training(
        self,
        train_data: Sequence[Any],
        labels: Sequence[str],
        active_features: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Processes tokenized elements, computes metrics, and fits the architecture."""
        rng = np.random.RandomState(
            self.seed
        )  # Creating an independent random state object

        active_features = self._normalize_active_features(active_features)
        baseline_features = self.discover_features(train_data)
        y = np.asarray(labels)

        is_dataframe = isinstance(train_data, pd.DataFrame)

        # Tracking sequential array row positional integers as to avoid bugs with indexing alignment
        positions = np.arange(len(train_data))

        holdout_metrics: Optional[dict[str, float]] = None
        holdout_report = pd.DataFrame()
        holdout_confusion_matrix = pd.DataFrame()

        cv_fold_metrics = pd.DataFrame()
        cv_mean_metrics: Optional[dict[str, float]] = None

        if self.run_holdout:
            # 1. Holdout Validation Partition Processing Step
            train_pos, test_pos = train_test_split(
                positions,
                test_size=self.test_size,
                random_state=rng,
                stratify=y,
                shuffle=True,
            )

            y_train = y[train_pos]
            y_test = y[test_pos]

            if is_dataframe:
                x_train_raw = train_data.iloc[train_pos]
                x_test_raw = train_data.iloc[test_pos]
            else:
                token_train = [
                    _tokenize_items(
                        [train_data[i]], include_bigrams=self.include_bigrams
                    )[0]
                    for i in train_pos
                ]
                token_test = [
                    _tokenize_items(
                        [train_data[i]], include_bigrams=self.include_bigrams
                    )[0]
                    for i in test_pos
                ]
                doc_labels_train = [f"train_doc_{i}" for i in train_pos]
                dtm_holdout = DTM()
                x_train_raw = dtm_holdout.fit_transform(
                    token_train, labels=list(doc_labels_train), min_df=self.min_df
                )
                x_test_raw = dtm_holdout.transform(token_test)

            if active_features is not None:
                # Use the DataFrame baseline if true, otherwise use the local DTM vocabulary
                holdout_baseline = (
                    baseline_features if is_dataframe else dtm_holdout.sorted_terms_list
                )

                x_train_raw = self._filter_active_features(
                    holdout_baseline, x_train_raw, active_features
                )
                x_test_raw = self._filter_active_features(
                    holdout_baseline, x_test_raw, active_features
                )

            scaler_holdout = StandardScaler(with_mean=False)
            x_train_scaled = scaler_holdout.fit_transform(x_train_raw)
            x_test_scaled = scaler_holdout.transform(x_test_raw)

            x_train_model, y_train_model = self._apply_smote(x_train_scaled, y_train)
            holdout_model = self._build_mlp_classifier()
            holdout_model.fit(x_train_model, y_train_model)

            y_pred = holdout_model.predict(_to_dense(x_test_scaled))
            holdout_metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            }

            report_dict = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            holdout_report = pd.DataFrame(report_dict).T

            class_labels = sorted(np.unique(y).tolist())
            cm = confusion_matrix(y_test, y_pred, labels=class_labels)
            holdout_confusion_matrix = pd.DataFrame(
                cm,
                index=[f"true_{l}" for l in class_labels],
                columns=[f"pred_{l}" for l in class_labels],
            )

        if self.run_cv:
            # 2. Stratified Cross Validation Cycle Sequence Block
            cv = StratifiedKFold(
                n_splits=self.cv_splits, shuffle=True, random_state=rng
            )
            cv_rows: list[dict[str, float]] = []

            cv_split_data = positions if is_dataframe else train_data
            for fold, (tr_idx, va_idx) in enumerate(
                cv.split(cv_split_data, y), start=1
            ):
                y_tr = y[tr_idx]
                y_va = y[va_idx]

                if is_dataframe:
                    x_tr_raw = train_data.iloc[tr_idx]
                    x_va_raw = train_data.iloc[va_idx]
                else:
                    fold_train_tokens = [
                        _tokenize_items(
                            [train_data[i]], include_bigrams=self.include_bigrams
                        )[0]
                        for i in tr_idx
                    ]
                    fold_valid_tokens = [
                        _tokenize_items(
                            [train_data[i]], include_bigrams=self.include_bigrams
                        )[0]
                        for i in va_idx
                    ]
                    fold_train_doc_labels = [f"fold_doc{i}" for i in tr_idx]
                    dtm_fold = DTM()
                    x_tr_raw = dtm_fold.fit_transform(
                        fold_train_tokens,
                        labels=list(fold_train_doc_labels),
                        min_df=self.min_df,
                    )
                    x_va_raw = dtm_fold.transform(fold_valid_tokens)

                if active_features is not None:
                    # Use the DataFrame baseline if true, otherwise use the local DTM vocabulary
                    cv_baseline = (
                        baseline_features
                        if is_dataframe
                        else dtm_fold.sorted_terms_list
                    )

                    x_tr_raw = self._filter_active_features(
                        cv_baseline, x_tr_raw, active_features
                    )
                    x_va_raw = self._filter_active_features(
                        cv_baseline, x_va_raw, active_features
                    )

                scaler_fold = StandardScaler(with_mean=False)
                x_tr_scaled = scaler_fold.fit_transform(x_tr_raw)
                x_va_scaled = scaler_fold.transform(x_va_raw)

                x_tr_model, y_tr_model = self._apply_smote(x_tr_scaled, y_tr)
                fold_model = self._build_mlp_classifier()
                fold_model.fit(x_tr_model, y_tr_model)

                fold_pred = fold_model.predict(_to_dense(x_va_scaled))
                cv_rows.append(
                    {
                        "fold": float(fold),
                        "accuracy": float(accuracy_score(y_va, fold_pred)),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(y_va, fold_pred)
                        ),
                        "macro_f1": float(f1_score(y_va, fold_pred, average="macro")),
                    }
                )  # End of for loop

            cv_fold_metrics = pd.DataFrame(cv_rows)
            cv_mean_metrics = {
                "accuracy": float(cv_fold_metrics["accuracy"].mean()),
                "balanced_accuracy": float(cv_fold_metrics["balanced_accuracy"].mean()),
                "macro_f1": float(cv_fold_metrics["macro_f1"].mean()),
            }

        # 3. Complete Corpus aggregation final model compilation
        dtm_final = None

        if is_dataframe:
            x_full_raw = train_data
        else:
            all_tokens_lists = _tokenize_items(
                train_data, include_bigrams=self.include_bigrams
            )
            all_doc_labels = [f"train_doc{i}" for i in range(len(all_tokens_lists))]
            dtm_final = DTM()
            x_full_raw = dtm_final.fit_transform(
                all_tokens_lists, labels=list(all_doc_labels), min_df=self.min_df
            )

        if active_features is not None:
            x_full_raw = self._filter_active_features(
                baseline_features, x_full_raw, active_features
            )

        scaler_final = StandardScaler(with_mean=False)
        x_full_scaled = scaler_final.fit_transform(x_full_raw)

        x_full_model, y_full_model = self._apply_smote(x_full_scaled, y)
        final_model = self._build_mlp_classifier()
        final_model.fit(x_full_model, y_full_model)

        final_pred = final_model.predict(_to_dense(x_full_scaled))
        final_metrics = {
            "accuracy": float(accuracy_score(y, final_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, final_pred)),
            "macro_f1": float(f1_score(y, final_pred, average="macro")),
        }

        final_report = pd.DataFrame(
            classification_report(y, final_pred, output_dict=True, zero_division=0)
        ).T

        return {
            "final_model": final_model,
            "final_dtm": dtm_final,
            "final_scaler": scaler_final,
            "final_metrics": final_metrics,
            "final_report": final_report,
            "holdout_metrics": holdout_metrics,
            "holdout_report": holdout_report,
            "holdout_confusion_matrix": holdout_confusion_matrix,
            "cv_fold_metrics": cv_fold_metrics,
            "cv_mean_metrics": cv_mean_metrics,
        }
