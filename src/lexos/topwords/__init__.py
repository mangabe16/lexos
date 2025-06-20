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

from typing import Any, Literal, Tuple  # Only keep Any and Literal

# register a custom extension for topwords if not already set
if not Doc.has_extension("topwords"):
    Doc.set_extension("topwords", default=None, force=True)
if not Doc.has_extension("keywords"):
    Doc.set_extension("keywords", default=None, force=True)


class TopwordsPlugin(BaseModel):
    """Base class for topwords plugins, providing a common API."""

    def to_df(self):
        """Return a pandas DataFrame representation of the model."""
        ...

class TextacyKeywords(TopwordsPlugin):
    """Extracts keywords from text or a spaCy Doc using textacy algorithms."""

    document: str | Doc | None = Field(None, description="The raw text or spaCy doc to analyze.")
    method: Literal["textrank", "sgrank"] = Field(
        ..., description="Method for keyword extraction (e.g., 'textrank', 'sgrank')."
    )
    topn: int = Field(
        10, gt=0, description="Number of top keywords to return."
    )
    model: str = Field(
        default="en_core_web_sm",
        description="spaCy model name to use for tokenization."
    )
    ngrams: Tuple[int, int] = Field(
        default=(1, 3),
        description="The ngram range for keyword extraction, e.g., (1, 1) for unigrams only."
    )
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    keywords: list[dict[str, Any]] | None = Field(
        default=None, description="Extracted keywords."
    )

    def __init__(self, **data):
        if "tokenizer" not in data or data["tokenizer"] is None:
            data["tokenizer"] = Tokenizer(model=data.get("model", "en_core_web_sm"))
        super().__init__(**data)

    def __call__(self) -> dict:
        if isinstance(self.document, Doc):
            doc = self.document
        elif isinstance(self.document, str):
            doc = self.tokenizer(self.document)
        else:
            raise ValueError("The 'document' field must be a string or a spaCy Doc.")

        min_n, max_n = self.ngrams

        if self.method == "textrank":
            results = extract.keyterms.textrank(
                doc, normalize="lemma", topn=self.topn * 20 # went for 20 instead of 3 to guarantee to get some words, might need to narrow it down later
            )
            # Filter by ngram length
            results = [
                (term, score)
                for term, score in results
                if min_n <= len(term.split()) <= max_n and term.lower() not in STOP_WORDS
            ][:self.topn]
        elif self.method == "sgrank":
            results = extract.keyterms.sgrank(
                doc, normalize="lower", ngrams=self.ngrams, topn=self.topn
            )
        else:
            raise ValueError("Invalid method. Choose 'textrank' or 'sgrank'.")

        self.keywords = [
            {"term": term, "score": score} for term, score in results
        ]
        doc._.keywords = self.keywords
        return self.to_dict()

    def to_dict(self):
        return {
            "keywords": [
                {"term": kw["term"], "score": kw["score"]}
                for kw in (self.keywords or [])
            ]
        }

    def to_df(self):
        return pd.DataFrame(getattr(self, "keywords", []))

    def to_list(self):
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
    ngrams: Tuple[int, int] = Field(
        default=(1, 1),
        description="The ngram range for analysis, e.g., (1, 1) for unigrams only."
    )
    model:str = Field(
        default="en_core_web_sm",
        description="spaCy model name to use for tokenization."
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

    def __init__(self, **data):
        #If tokenizer is not provided, create one with the specified model
        if "tokenizer" not in data or data["tokenizer"] is None:
            data["tokenizer"] = Tokenizer(model=data.get("model","en_core_web_sm"))
        super().__init__(**data)

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

        def get_ngrams(doc, n):
            tokens = [
                token.lower_ if not self.case_sensitive else token.text
                for token in doc
                if not (
                    (self.remove_stopwords and token.is_stop) or
                    (self.remove_punct and token.is_punct) or
                    (self.remove_digits and token.is_digit) or
                    token.is_space
                )
            ]
            return [
                " ".join(tokens[i:i+n])
                for i in range(len(tokens) - n + 1)
            ]

        def get_tokens(docs: list[Any]) -> list[str]:
            tokens: list[str] = []
            min_n, max_n = self.ngrams
            for doc in docs:
                for n in range(min_n, max_n + 1):
                    tokens.extend(get_ngrams(doc, n))
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
        return pd.DataFrame(getattr(self, "topwords", []) or [], columns=["term", "z_score"])

    def to_list(self):
        """Return the topwords as a list of (term, z_score) tuples."""
        return getattr(self, "topwords", [])