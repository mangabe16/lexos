"""mlp_pipeline.py.

A leakage-safe Multi-Layer Perceptron Text Classification Pipeline Strategy
complying with the Lexos Strategy Pattern specification.

Last Updated: June 22, 2026
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


# Utility compilation helpers left decoupled for architectural performance mapping
def _tokenize_items(
    items: Sequence[Any], include_bigrams: bool = True
) -> list[list[str]]:
    """Transform sequence objects or strings into normal unigram/bigram token vectors."""
    ws_tokenizer = WhitespaceTokenizer()
    ngrams = Ngrams(n=2)
    token_lists: list[list[str]] = []

    for item in items:
        # Check if incoming item is a spacy Doc or raw string object
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
    min_df: int = Field(default=2, description="Minimum data document expression limit")
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

    def _apply_smote(
        self, feature_matrix: Any, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling safely across dense arrays."""
        dense_matrix = _to_dense(feature_matrix)
        if not self.use_smote:
            return dense_matrix, labels

        if SMOTE is None:
            raise ImportError(
                "SMOTE requested, but imbalanced-learn is not installed. "
                "Install it with `pip install imbalanced-learn` or set `use_smote=False`."
            )
        smote = SMOTE(random_state=self.seed)
        return smote.fit_resample(dense_matrix, labels)

    def _build_mlp_classifier(self) -> MLPClassifier:
        """Instantiate an explicit MLP Classifier using current keyword parameters."""
        kwargs = dict(self.mlp_kwargs)
        kwargs.setdefault("random_state", self.seed)
        return MLPClassifier(**kwargs)

    def execute_training(
        self, train_data: Sequence[Any], labels: Sequence[str]
    ) -> dict[str, Any]:
        """Processes tokenized elements, computes metrics, and fits the architecture."""
        np.random.seed(self.seed)

        all_token_lists = _tokenize_items(
            train_data, include_bigrams=self.include_bigrams
        )
        all_doc_labels = [f"train_doc_{i}" for i in range(len(all_token_lists))]
        y = np.asarray(labels)
        indices = np.arange(len(all_token_lists))

        # 1. Holdout Validation Partition Processing Step
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y,
            shuffle=True,
        )

        token_train = [all_token_lists[i] for i in train_idx]
        token_test = [all_token_lists[i] for i in test_idx]
        doc_labels_train = [all_doc_labels[i] for i in train_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        dtm_holdout = DTM()
        x_train = dtm_holdout.fit_transform(
            token_train, labels=list(doc_labels_train), min_df=self.min_df
        )
        x_test = dtm_holdout.transform(token_test)

        scaler_holdout = StandardScaler(with_mean=False)
        x_train_scaled = scaler_holdout.fit_transform(x_train)
        x_test_scaled = scaler_holdout.transform(x_test)

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

        # 2. Stratified Cross Validation Cycle Sequence Block
        cv = StratifiedKFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.seed
        )
        cv_rows: list[dict[str, float]] = []

        for fold, (tr_idx, va_idx) in enumerate(cv.split(all_token_lists, y), start=1):
            fold_train_tokens = [all_token_lists[i] for i in tr_idx]
            fold_valid_tokens = [all_token_lists[i] for i in va_idx]
            fold_train_doc_labels = [all_doc_labels[i] for i in tr_idx]
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            dtm_fold = DTM()
            x_tr = dtm_fold.fit_transform(
                fold_train_tokens,
                labels=list(fold_train_doc_labels),
                min_df=self.min_df,
            )
            x_va = dtm_fold.transform(fold_valid_tokens)

            scaler_fold = StandardScaler(with_mean=False)
            x_tr_scaled = scaler_fold.fit_transform(x_tr)
            x_va_scaled = scaler_fold.transform(x_va)

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
            )

        cv_fold_metrics = pd.DataFrame(cv_rows)
        cv_mean_metrics = {
            "accuracy": float(cv_fold_metrics["accuracy"].mean()),
            "balanced_accuracy": float(cv_fold_metrics["balanced_accuracy"].mean()),
            "macro_f1": float(cv_fold_metrics["macro_f1"].mean()),
        }

        # 3. Complete Corpus Aggregation Final Model Compilation
        dtm_final = DTM()
        x_full = dtm_final.fit_transform(
            all_token_lists, labels=list(all_doc_labels), min_df=self.min_df
        )

        scaler_final = StandardScaler(with_mean=False)
        x_full_scaled = scaler_final.fit_transform(x_full)

        x_full_model, y_full_model = self._apply_smote(x_full_scaled, y)
        final_model = self._build_mlp_classifier()
        final_model.fit(x_full_model, y_full_model)

        return {
            "final_model": final_model,
            "final_dtm": dtm_final,
            "final_scaler": scaler_final,
            "holdout_metrics": holdout_metrics,
            "holdout_report": holdout_report,
            "holdout_confusion_matrix": holdout_confusion_matrix,
            "cv_fold_metrics": cv_fold_metrics,
            "cv_mean_metrics": cv_mean_metrics,
        }
