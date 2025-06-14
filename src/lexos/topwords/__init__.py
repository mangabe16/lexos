from collections import Counter
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

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
    """Extracts keywords from text using textacy algorithms."""

    text: str = Field(..., description="The raw text to analyze.")
    method: Literal["textrank", "sgrank"] = Field(
        "textrank", description="The keyword extraction method." # Defaults to "textrank"
    )
    topn: int = Field(10, gt=0, description="Number of top keywords to return.") # Defaults to 10, must be > 0
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract keywords from the input text using the specified method.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Extracted keywords and their scores.
        """
        doc = self.tokenizer.make_doc(self.text) # process the raw input text into a spaCy Doc object.

        if self.method == "textrank":
            results: List[Tuple[str, float]] = extract.keyterms.textrank(doc, normalize="lemma", topn=self.topn)
        elif self.method == "sgrank":
            results: List[Tuple[str, float]] = extract.keyterms.sgrank(
                doc, normalize="lower", ngrams=(1, 2, 3), topn=self.topn
            )
        else:
            raise ValueError("Invalid keyword extraction method.")

        keywords_list = [{"term": term, "score": score} for term, score in results] # Format results for consistent output
        doc._.keywords = keywords_list
        self.doc = doc # storing the spaCy doc so it can be accessed later

    def to_dict(self):
        """Return the extracted keywords as a dictionary."""
        return {"keywords": getattr(self, "keywords", [])}

    def to_df(self):
        """Return the extracted keywords as a pandas DataFrame."""
        return pd.DataFrame(getattr(self, "keywords", []))

    def to_list(self):
        """Return the extracted keywords as a list of (term, score) tuples."""
        return [(kw["term"], kw["score"]) for kw in getattr(self, "keywords", [])]

class ZTestTopwords(TopwordsPlugin):
    """Calculates top distinguishing words using Z-test for significance."""

    target_texts: List[str] = Field(..., description="List of target documents.")
    background_texts: List[str] = Field(..., description="List of background documents.")
    topn: int = Field(10, gt=0, description="Number of top words to return.")
    case_sensitive: Optional[bool] = Field(True, description="Whether analysis is case sensitive.")
    remove_stopwords: Optional[bool] = Field(True, description="Whether to remove stopwords.")
    remove_punct: Optional[bool] = Field(True, description="Whether to remove punctuation.")
    remove_digits: Optional[bool] = Field(False, description="Whether to remove digits.")
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)
    docs: Optional[List[Any]] = Field(None, description="Optional list of spaCy Doc objects.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self) -> Dict[str, Any]:
        """
        Calculate top distinguishing words using Z-test for significance.

        Returns:
            Dict[str, Any]: Top words and their Z-scores.
        """
        target_docs: List[Any] = list(self.tokenizer.make_docs(self.target_texts))
        background_docs: List[Any] = list(self.tokenizer.make_docs(self.background_texts))

        def get_tokens(docs: List[Any]) -> List[str]:
            """
            Extract tokens from a list of spaCy Doc objects, applying filters.

            Args:
                docs (List[Any]): List of spaCy Doc objects.

            Returns:
                List[str]: List of filtered tokens.
            """
            tokens = []
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

        target_tokens: List[str] = get_tokens(target_docs)
        background_tokens: List[str] = get_tokens(background_docs)

        target_counts: Counter = Counter(target_tokens)
        background_counts: Counter = Counter(background_tokens)

        target_total: int = sum(target_counts.values())
        background_total: int = sum(background_counts.values())

        results: List[Tuple[str, float]] = []
        all_terms: set = set(target_counts) | set(background_counts)
        for term in all_terms:
            p1: float = target_counts[term] / target_total if target_total else 0
            p2: float = background_counts[term] / background_total if background_total else 0
            p: float = (target_counts[term] + background_counts[term]) / (target_total + background_total) if (target_total + background_total) else 0
            n1, n2 = target_total, background_total

            if n1 > 0 and n2 > 0 and p > 0 and p < 1:
                denominator = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
                if denominator != 0:
                    z: float = (p1 - p2) / denominator
                else:
                    z = 0.0
            else:
                z = 0.0

            results.append((term, z))

        sorted_results = sorted(results, key=lambda item: abs(item[1]), reverse=True)
        topwords = sorted_results[:self.topn]

        # if docs are provided, set the topwords attribute on each
        if self.docs is not None:
            for doc in self.docs:
                doc._.topwords = topwords
    
    def to_dict(self):
        """Return the topwords as a dictionary with terms and Z-scores."""
        return {"topwords": [{"term": term, "z_score": z_score} for term, z_score in getattr(self, "topwords", [])]}

    def to_df(self):
        """Return the topwords as a pandas DataFrame."""
        return pd.DataFrame(getattr(self, "topwords", []), columns=["term", "z_score"])
    
    def to_list(self):
        """Return the topwords as a list of (term, z_score) tuples."""
        return getattr(self, "topwords", [])