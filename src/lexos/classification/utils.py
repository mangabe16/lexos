"""Shared utilities for Lexos classification pipelines."""

from typing import Any, Sequence

import numpy as np
import pandas as pd

from lexos.tokenizer import WhitespaceTokenizer
from lexos.tokenizer.ngrams import Ngrams


def _tokenize_items(
    items: Sequence[Any], include_bigrams: bool = True
) -> list[list[str]]:
    """Transform sequence objects or strings into unigram/bigram token vectors."""
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


def save_predictions(filenames: list, predictions: list, output_file: str) -> None:
    """Save filenames and predicted labels to a CSV file."""
    df = pd.DataFrame({"filename": filenames, "prediction": predictions})
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


class PredictionSaver:
    """Simple wrapper class to save predictions (kept for API compatibility)."""

    def __init__(self, default_output: str = "predictions.csv"):
        """Initialize the saver with a default output path."""
        self.default_output = default_output

    def save(
        self,
        filenames: Sequence[str],
        predictions: Sequence[str],
        output_file: str | None = None,
    ) -> None:
        """Save filenames and predictions to the selected output path."""
        target = output_file or self.default_output
        save_predictions(list(filenames), list(predictions), target)


__all__ = [
    "_tokenize_items",
    "_to_dense",
    "save_predictions",
    "PredictionSaver",
]
