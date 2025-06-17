from collections import Counter

import numpy as np
import pandas as pd
import spacy
from pydantic import BaseModel, Field, ValidationError
from pydantic.config import ConfigDict
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc
import textacy
from textacy import extract
from lexos.tokenizer import Tokenizer

from typing import Any, Literal

# register a custom extension for topwords if not already set
if not Doc.has_extension("topwords"):
    Doc.set_extension("topwords", default=None, force=True)
if not Doc.has_extension("keywords"):
    Doc.set_extension("keywords", default=None, force=True)


class TopwordsPlugin(BaseModel):
    """Base class for topwords plugins, providing a common API."""

    def to_dict(self):
        """Return a dictionary representation of the model."""
        return self.model_dump()


class TextacyKeywords(TopwordsPlugin):
    """Extracts keywords from text or a spaCy Doc using textacy algorithms."""

    document: str | Doc | None = Field(None, description="The raw text or spaCy doc to analyze.")
    method: Literal["textrank", "sgrank"] = Field(
        ..., description="Method for keyword extraction (e.g., 'textrank', 'sgrank')."
    )
    topn: int = Field(
        10, gt=0, description="Number of top keywords to return."
    )
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    keywords: list[dict[str, Any]] | None = Field(
        default=None, description="Extracted keywords."
    )

    def __call__(self) -> dict:
        """
        Extract keywords from the input document (string or spaCy Doc) using the specified method.
        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted keywords and their scores.
        """
        if isinstance(self.document, Doc):
            doc = self.document
        elif isinstance(self.document, str):
            doc = self.tokenizer.make_doc(self.document)
        else:
            raise ValueError("The 'document' field must be a string or a spaCy Doc.")

        if self.method == "textrank":
            results: list[tuple[str, float]] = extract.keyterms.textrank(
                doc, normalize="lemma", topn=self.topn
            )
        elif self.method == "sgrank":
            results: list[tuple[str, float]] = extract.keyterms.sgrank(
                doc, normalize="lower", ngrams=(1, 2, 3), topn=self.topn
            )

        self.keywords = [
            {"term": term, "score": score} for term, score in results
        ]
        doc._.keywords = self.keywords
        return self.to_dict()

    def to_dict(self):
        """Return the keywords as a dictionary with terms and scores."""
        data = super().to_dict()
        data["keywords"] = [
            {"term": term, "score": score}
            for term, score in getattr(self, "keywords", [])
        ]
        return data

    def to_df(self):
        """Return the extracted keywords as a pandas DataFrame."""
        return pd.DataFrame(getattr(self, "keywords", []))

    def to_list(self):
        """Return the extracted keywords as a list of (term, score) tuples."""
        return [(kw["term"], kw["score"]) for kw in getattr(self, "keywords", [])]


class ZTestTopwords(TopwordsPlugin):
    """Calculates top distinguishing words using Z-test for significance."""

    target_documents: list[str | Doc] | None = Field(
        None, description="List of target documents, either strings or spaCy docs"
    )
    background_documents: list[str | Doc] | None = Field(
        None, description="List of background documents, either strings or spaCy docs"
    )
    topn: int = Field(10, gt=0, description="Number of top words to return.")
    case_sensitive: bool | None = Field(
        True, description="Whether analysis is case sensitive."
    )
    remove_stopwords: bool | None = Field(
        True, description="Whether to remove stopwords."
    )
    remove_punct: bool | None = Field(
        True, description="Whether to remove punctuation."
    )
    remove_digits: bool | None = Field(
        False, description="Whether to remove digits."
    )
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)
    docs: list[Any] | None = Field(
        None, description="Optional list of spaCy Doc objects to set results on."
    )
    topwords: list[tuple[str, float]] | None = Field(
        default=None, description="Top distinguished words."
    )
    output_format: str = Field("dict", description="Output format: dict, dataframe, list_of_dicts, or list_of_tuples")
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self) -> dict:
        """
        Calculate top distinguishing words using Z-test for significance.

        Returns:
            Dict[str, Any]: Top words and their Z-scores.
        """
        # Use provided docs or create them from text
        if self.target_documents is not None:
            target_docs = [
                doc if isinstance(doc, Doc) else self.tokenizer.make_doc(doc)
                for doc in self.target_documents
            ]
        else:
            raise ValueError("The 'target_documents' field must be provided.")

        if self.background_documents is not None:
            background_docs = [
                doc if isinstance(doc, Doc) else self.tokenizer.make_doc(doc)
                for doc in self.background_documents
            ]
        else:
            raise ValueError("The 'background_documents' field must be provided.")

        def get_tokens(docs: list[Any]) -> list[str]:
            """
            Extract tokens from a list of spaCy Doc objects, applying filters.

            Args:
                docs (list[Any]): List of spaCy Doc objects.

            Returns:
                list[str]: List of filtered tokens.
            """
            tokens: list[str] = []
            for doc in docs:
                for token in doc:
                    if not self.case_sensitive:
                        token_text = token.lower_
                    else:
                        token_text = token.text

                    if self.remove_stopwords and token.is_stop:
                        continue
                    if self.remove_punct and token.is_punct:
                        continue
                    if self.remove_digits and token.is_digit:
                        continue
                    if not token.is_stop and not token.is_punct and not token.is_space:
                        tokens.append(token_text)
            return tokens

        target_tokens: list[str] = get_tokens(target_docs)
        background_tokens: list[str] = get_tokens(background_docs)

        target_counts: Counter = Counter(target_tokens)
        background_counts: Counter = Counter(background_tokens)

        target_total: int = sum(target_counts.values())
        background_total: int = sum(background_counts.values())

        results: list[tuple[str, float]] = []
        all_terms: set = set(target_counts) | set(background_counts)
        for term in all_terms:
            p1: float = target_counts[term] / target_total if target_total else 0
            p2: float = (
                background_counts[term] / background_total if background_total else 0
            )
            p: float = (
                (target_counts[term] + background_counts[term])
                / (target_total + background_total)
                if (target_total + background_total)
                else 0
            )
            n1, n2 = target_total, background_total

            if n1 > 0 and n2 > 0 and p > 0 and p < 1:
                denominator = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
                if denominator != 0:
                    z: float = (p1 - p2) / denominator
                else:
                    z = 0.0
            else:
                z = 0.0

            results.append((term, z))

        # Filter out terms with a Z-score of 0.0 before sorting.
        filtered_results = [item for item in results if item[1] != 0.0]

        sorted_results = sorted(
            filtered_results, key=lambda item: abs(item[1]), reverse=True
        )
        self.topwords = sorted_results[: self.topn]

        # if docs are provided, set the topwords attribute on each
        if self.docs is not None:
            for doc in self.docs:
                doc._.topwords = self.topwords

        # Output format logic
        if self.output_format == "dict":
            return self.to_dict()
        elif self.output_format == "dataframe":
            return {"topwords_df": self.to_df()}
        elif self.output_format == "list_of_dicts":
            return {"topwords_list": [
                {"term": term, "z_score": z_score} for term, z_score in self.topwords
            ]}
        elif self.output_format == "list_of_tuples":
            return {"topwords_list": self.to_list()}
        else:
            raise ValueError(f"Invalid output_format: {self.output_format}")

    def to_dict(self):
        """Return the topwords as a dictionary with terms and Z-scores."""
        return {
            "topwords":
            [
                {"term": term, "z_score": z_score}
                for term, z_score in getattr(self, "topwords", [])
            ]
        }

    def to_df(self):
        """Return the topwords as a pandas DataFrame."""
        return pd.DataFrame(getattr(self, "topwords", []), columns=["term", "z_score"])

    def to_list(self):
        """Return the topwords as a list of (term, z_score)