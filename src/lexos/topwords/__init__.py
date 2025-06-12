from typing import Optional, Literal, List, Dict, Any, Tuple, Iterable
from pydantic import BaseModel, Field, ValidationError
from pydantic.config import ConfigDict
import textacy
from textacy import extract
from lexos.tokenizer import Tokenizer
from collections import Counter
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


class TextacyKeywords(BaseModel):
    """Extracts keywords from text using textacy algorithms."""

    text: str = Field(..., description="The raw text to analyze.")
    method: Literal["textrank", "sgrank"] = Field(
        "textrank", description="The keyword extraction method."
    )
    topn: int = Field(10, gt=0, description="Number of top keywords to return.")

    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self) -> Dict[str, List[Dict[str, Any]]]:
        doc = self.tokenizer.make_doc(self.text)

        if self.method == "textrank":
            results: List[Tuple[str, float]] = extract.keyterms.textrank(doc, normalize="lemma", topn=self.topn)
        elif self.method == "sgrank":
            results: List[Tuple[str, float]] = extract.keyterms.sgrank(
                doc, normalize="lower", ngrams=(1, 2, 3), topn=self.topn
            )
        # MODIFIED: The else block was removed as it's made unreachable by Pydantic's validation.

        keywords_list = [{"term": term, "score": score} for term, score in results]
        return {"keywords": keywords_list}


class ZTestTopwords(BaseModel):
    """Calculates top distinguishing words using Z-test for significance."""

    target_texts: List[str] = Field(..., description="List of target documents.")
    background_texts: List[str] = Field(..., description="List of background documents.")
    topn: int = Field(10, gt=0, description="Number of top words to return.")
    case_sensitive: Optional[bool] = Field(True, description="Whether analysis is case sensitive.")
    remove_stopwords: Optional[bool] = Field(True, description="Whether to remove stopwords.")
    remove_punct: Optional[bool] = Field(True, description="Whether to remove punctuation.")
    remove_digits: Optional[bool] = Field(False, description="Whether to remove digits.")

    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self) -> Dict[str, List[Dict[str, Any]]]:
        target_docs: List[Any] = list(self.tokenizer.make_docs(self.target_texts))
        background_docs: List[Any] = list(self.tokenizer.make_docs(self.background_texts))

        def get_tokens(docs: List[Any]) -> List[str]:
            tokens = []
            for doc in docs:
                for token in doc:
                    token_lower = token.text.lower() # Get lowercased text for stopword check
                    # Conditions for SKIPPING a token
                    if token.is_space:
                        continue
                    # MODIFIED: Use the imported STOP_WORDS list for a reliable check.
                    if self.remove_stopwords and token_lower in STOP_WORDS:
                        continue
                    if self.remove_punct and token.is_punct:
                        continue
                    if self.remove_digits and token.is_digit:
                        continue

                    # If token is not skipped, append it.
                    if not self.case_sensitive:
                        tokens.append(token_lower)
                    else:
                        tokens.append(token.text)
            return tokens

        target_tokens: List[str] = get_tokens(target_docs)
        background_tokens: List[str] = get_tokens(background_docs)

        target_counts: Counter = Counter(target_tokens)
        background_counts: Counter = Counter(background_tokens)

        target_total: int = sum(target_counts.values())
        background_total: int = sum(background_counts.values())

        results: List[Tuple[str, float]] = []
        all_terms: set = set(target_counts) | set(background_counts)

        if target_total == 0 or background_total == 0:
            return {"topwords": []}

        for term in all_terms:
            p1: float = target_counts[term] / target_total
            p2: float = background_counts[term] / background_total
            p: float = (target_counts[term] + background_counts[term]) / (target_total + background_total)
            n1, n2 = target_total, background_total

            if p > 0 and p < 1:
                denominator = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
                z: float = (p1 - p2) / denominator if denominator != 0 else 0.0
            else:
                z = 0.0

            results.append((term, z))

        sorted_results = sorted(results, key=lambda item: abs(item[1]), reverse=True)
        non_zero_results = [item for item in sorted_results if item[1] != 0.0]

        top_words_list = [{"term": term, "z_score": z_score} for term, z_score in non_zero_results[:self.topn]]

        return {"topwords": top_words_list}