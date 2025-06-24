from lexos.topwords import TopWords
from spacy.tokens import Doc
from pydantic import Field, ConfigDict
from spacy.schemas import DocJSONSchema
from lexos.tokenizer import Tokenizer
import numpy as np
from collections import Counter
from typing import Any
import pandas as pd
from lexos.topwords.comparison_handler import ComparisonHandler


validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
)

# register a custom extension for topwords if not already set
if not Doc.has_extension("topwords"):
    Doc.set_extension("topwords", default=None, force=True)


class ZTest(TopWords):
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
    remove_digits: bool | None = Field(False, description="Whether to remove digits.")
    ngrams: tuple[int, int] = Field(
        default=(1, 1),
        description="The ngram range for analysis, e.g., (1, 1) for unigrams only.",
    )
    model: str = Field(
        default="xx_sent_ud_sm",
        description="spaCy model name to use for tokenization.",
    )
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)
    docs: list[Any] | None = Field(
        None, description="Optional list of spaCy Doc objects to set results on."
    )
    topwords: list[tuple[str, float]] | None = Field(
        default=None, description="Top distinguished words."
    )
    output_format: str = Field(
        "dict",
        description="Output format: dict, dataframe, list_of_dicts, or list_of_tuples",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_config = validation_config

    def __init__(self, **data):
        """Initialize the ZTest class, ensuring a tokenizer is set.

        If a tokenizer is not provided, creates one using the specified spaCy model.
        """
        # If tokenizer is not provided, create one with the specified model
        if "tokenizer" not in data or data["tokenizer"] is None:
            data["tokenizer"] = Tokenizer(model=data.get("model", "xx_sent_ud_sm"))
        super().__init__(**data)

    def _get_ngrams(self, doc, n):
        """Generate n-grams from a spaCy Doc after applying preprocessing filters.

        Args:
            doc (spacy.tokens.Doc): The spaCy Doc to extract n-grams from.
            n (int): The length of n-grams to generate.

        Returns:
            list[str]: List of n-gram strings.
        """
        tokens = [
            token.lower_ if not self.case_sensitive else token.text
            for token in doc
            if not (
                (self.remove_stopwords and token.is_stop)
                or (self.remove_punct and token.is_punct)
                or (self.remove_digits and token.is_digit)
                or token.is_space
            )
        ]
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _get_tokens(self, docs: list[Any]) -> list[str]:
        """Extract all n-grams from a list of spaCy Docs according to the configured ngram range.

        Args:
            docs (list): List of spaCy Doc objects.

        Returns:
            list[str]: List of all n-grams from all documents.
        """
        tokens: list[str] = []
        min_n, max_n = self.ngrams
        for doc in docs:
            for n in range(min_n, max_n + 1):
                tokens.extend(self._get_ngrams(doc, n))
        return tokens

    def __call__(self) -> dict:
        """Calculate top distinguishing words using Z-test for significance.

        Returns:
            dict[str, Any]: Top words and their Z-scores.
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

        target_tokens: list[str] = self._get_tokens(target_docs)
        background_tokens: list[str] = self._get_tokens(background_docs)

        target_counts: Counter = Counter(target_tokens)
        background_counts: Counter = Counter(background_tokens)

        target_total: int = sum(target_counts.values())
        background_total: int = sum(background_counts.values())

        results: list[tuple[str, float]] = []
        all_terms: set = set(target_counts) | set(background_counts)
        for term in all_terms:
            # Proportion of this term in target and background
            p1: float = target_counts[term] / target_total if target_total else 0
            p2: float = (
                background_counts[term] / background_total if background_total else 0
            )
            # Combined proportion of this term in both sets
            p: float = (
                (target_counts[term] + background_counts[term])
                / (target_total + background_total)
                if (target_total + background_total)
                else 0
            )
            n1, n2 = target_total, background_total

            # Only calculate Z if both sets have data and p is not 0 or 1
            if n1 > 0 and n2 > 0 and p > 0 and p < 1:
                # Standard error for the difference in proportions
                denominator = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
                if denominator != 0:
                    # Z-score: difference in proportions divided by standard error
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
            return self.to_df()  # <--- Direct return
        elif self.output_format == "list_of_dicts":
            return self.to_list_of_dicts()  # <--- Direct return
        elif self.output_format == "list_of_tuples":
            return self.to_list()  # <--- Direct return
        else:
            raise ValueError(f"Invalid output_format: {self.output_format}")

    def to_dict(self):
        """Return the topwords as a dictionary with terms and Z-scores."""
        return {
            "topwords": [
                {"term": term, "z_score": z_score}
                for term, z_score in getattr(self, "topwords", [])
            ]
        }

    def to_df(self):
        """Return the topwords as a pandas DataFrame."""
        return pd.DataFrame(
            getattr(self, "topwords", []) or [], columns=["term", "z_score"]
        )

    def to_list_of_dicts(self):
        """Return the topwords as a list of dictionaries with 'term' and 'z_score'."""
        return [
            {"term": term, "z_score": z_score}
            for term, z_score in getattr(self, "topwords", [])
        ]

    def to_list(self):
        """Return the topwords as a list of (term, z_score) tuples."""
        return getattr(self, "topwords", [])
