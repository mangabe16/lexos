"""ZTest.py.

Last Updated: June 25, 2025
Last Tested: June 25, 2025
"""

from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

"""ZTest.py.

Last Updated: June 25, 2025
Last Tested: June 25, 2025
"""

from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc

from lexos.tokenizer import Tokenizer
from lexos.topwords import TopWords
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

    def __call__(self) -> dict | list | pd.DataFrame | list[dict] | list[tuple]:
        """Calculate top distinguishing words using Z-test for significance.

        Returns:
            dict | list | pd.DataFrame | list[dict] | list[tuple]: Top words and their Z-scores.
            dict | list | pd.DataFrame | list[dict] | list[tuple]: Top words and their Z-scores.
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


class ZTestComparison(BaseModel):
    """Unified comparison handler for Z-test operations on documents and categories."""

    corpus: list[Doc | str] = Field(..., description="Full corpus of documents")
    labels: Optional[list[str]] = Field(
        None, description="Optional labels for documents"
    )
    cats: Optional[list[str]] = Field(None, description="Categories for each document")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    output_format: str = Field(
        "dict",
        description="Output format: 'dict, 'dataframe', 'list_of_dicts', 'list_of_tuples'",
    )

    def model_post_init(self, __context):
        """Validate and initialize labels after model initialization."""
        if not self.corpus or len(self.corpus) < 2:
            raise ValueError("Corpus must contain at least 2 documents for comparison.")

        # Generate default labels if not provided
        if self.labels is None:
            object.__setattr__(
                self, "labels", [f"Doc_{i + 1}" for i in range(len(self.corpus))]
            )

        if self.labels is not None and len(self.labels) != len(self.corpus):
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must match corpus length ({len(self.corpus)})."
            )

        # Validate categories if provided
        if self.cats is not None and len(self.cats) != len(self.corpus):
            raise ValueError(
                f"Number of categories ({len(self.cats)}) must match corpus length ({len(self.corpus)})."
            )

    def compare_docs_to_corpus(
        self,
        docs: list[Doc | str],
        background_cats: Optional[list[str]] = None,
        **kwargs,
    ) -> list[dict[str, str | ZTest]]:
        """Compare each document against a filtered background (by categories if specified).

        Args:
            docs: Documents to compare against the corpus. Must be a subset of corpus.
            **kwargs: Additional arguments passed to ZTest constructor.

        Returns:
            List of dictionaries, each containing:
            - "label": Document label
            - "result": ZTest instance with comparison results

        Raises:
            ValueError: If docs is not a subset of corpus.
        """
        if not docs:
            raise ValueError("Must provide at least one document to compare.")

        # Get labels for the specified docs
        doc_labels = self._get_doc_labels(docs)
        results = []

        for label, doc in zip(doc_labels, docs):
            # build background set
            if background_cats is not None and self.cats is not None:
                background_docs = [
                    d
                    for d, c in zip(self.corpus, self.cats)
                    if c in background_cats and d != doc
                ]
            else:
                background_docs = [d for d in self.corpus if d != doc]

            if not background_docs:
                raise ValueError("No background documents available for comparison.")

            ztest_instance = ZTest(
                target_documents=[doc], background_documents=background_docs, **kwargs
            )
            results.append({"label": label, "result": ztest_instance})

        return results

    def compare_cat_to_corpus(
        self, cat: str, background_cats: Optional[list[str]] = None, **kwargs
    ) -> dict[str, str | list[str] | ZTest]:
        """Compare documents in a category against other specified categories.

        Args:
            cat: The category to compare against others.
            background_cats: Optional list of categories to use as background.
                           If None, uses all categories except the target category.
            **kwargs: Additional arguments passed to ZTest constructor.

        Returns:
            Dictionary containing:
            - "category": The category being compared
            - "background_categories": List of categories used as background
            - "result": ZTest instance with comparison results

        Raises:
            ValueError: If categories are not provided or invalid.
        """
        if self.cats is None:
            raise ValueError(
                "Categories (cats) must be provided for category comparison."
            )

        # Get available categories and validate target
        available_cats = set(self.cats)
        if cat not in available_cats:
            sorted_cats = sorted(available_cats)
            raise ValueError(
                f"Category '{cat}' not found in cats. Available categories: {sorted_cats}"
            )

        # Determine background categories
        background_categories = self._determine_background_categories(
            cat, background_cats, available_cats
        )

        # Separate documents by category
        target_documents, background_documents = self._separate_docs_by_category(
            cat, background_categories
        )

        # Create ZTest instance
        ztest_instance = ZTest(
            target_documents=target_documents,
            background_documents=background_documents,
            **kwargs,
        )

        return {
            "category": cat,
            "background_categories": background_categories,
            "result": ztest_instance,
        }

    def compare_all_cats_to_corpus(
        self, background_cats: Optional[list[str]] = None, **kwargs
    ) -> dict[str, dict[str, str | list[str] | ZTest]]:
        """Compare each category against specified background categories.

        Args:
            background_cats: Optional list of categories to use as background for each comparison.
                           If None, for each category, uses all other categories as background.
            **kwargs: Additional arguments passed to ZTest constructor.

        Returns:
            Dictionary mapping category names to comparison results.
        """
        if self.cats is None:
            raise ValueError(
                "Categories (cats) must be provided for category comparison."
            )

        available_cats = set(self.cats)
        results = {}

        for cat in sorted(available_cats):
            # For each category, determine its specific background categories
            if background_cats is None:
                cat_background_cats = None  # Will use all others
            else:
                # Remove current category from background if it's there
                cat_background_cats = [c for c in background_cats if c != cat]
                if not cat_background_cats:
                    continue  # Skip if no valid background categories

            try:
                results[cat] = self.compare_cat_to_corpus(
                    cat=cat, background_cats=cat_background_cats, **kwargs
                )
            except ValueError as e:
                # Log warning but continue with other categories
                print(f"Warning: Skipping category '{cat}': {e}")
                continue

        return results

    def compare_each_doc_to_other_classes(self, **kwargs) -> dict[str, list[dict]]:
        """For each class, compare each document in that class to all documentes in otehr classes.

        Returns a dict mapping class name to a list of results for each document in that class.
        """
        if self.cats is None:
            raise ValueError("Categories (cats) must be provided for this comparison")

        # Build mapping from class to docs
        class_to_docs = {}
        for doc, cat in zip(self.corpus, self.cats):
            if cat not in class_to_docs:
                class_to_docs[cat] = []
            class_to_docs[cat].append(doc)

        results = {}
        for cat, docs_in_class in class_to_docs.items():
            # Background: all docs not in this class
            background_docs = [
                doc for doc, c in zip(self.corpus, self.cats) if c != cat
            ]
            results[cat] = []
            for doc in docs_in_class:
                ztest_instance = ZTest(
                    target_documents=[doc],
                    background_documents=background_docs,
                    **kwargs,
                )
                label = self._get_doc_labels([doc])[0]
                results[cat].append({"label": label, "result": ztest_instance})
        return results

    def compare_each_class_to_each_other_class(
        self, **kwargs
    ) -> dict[str, dict[str, dict]]:
        """For each class, compare it to every other class individually.

        Returns a nested dict: {target_class: {background_class: result_dict}}
        """
        if self.cats is None:
            raise ValueError("Categories (cats) must be provided for this comparison.")

        # build mapping from class to docs
        class_to_docs = {}
        for doc, cat in zip(self.corpus, self.cats):
            if cat not in class_to_docs:
                class_to_docs[cat] = []
            class_to_docs[cat].append(doc)

        results = {}
        all_classes = list(class_to_docs.keys())
        for target_class in all_classes:
            results[target_class] = {}
            target_docs = class_to_docs[target_class]
            for background_class in all_classes:
                if background_class == target_class:
                    continue
                background_docs = class_to_docs[background_class]
                if not target_docs or not background_docs:
                    continue
                ztest_instance = ZTest(
                    target_documents=target_docs,
                    background_documents=background_docs,
                    **kwargs,
                )
                results[target_class][background_class] = {
                    "target_category": target_class,
                    "background_category": background_class,
                    "result": ztest_instance,
                }
        return results

    # Helper methods
    def _get_doc_labels(self, docs: list[Doc | str]) -> list[str]:
        """Get labels for specific documents."""
        if not docs:
            return []

        # Find indices of docs in corpus
        corpus_strs = [str(doc) for doc in self.corpus]
        doc_labels = []

        # Ensure labels are initialized
        if self.labels is None:
            self.labels = [f"Doc_{i + 1}" for i in range(len(self.corpus))]

        for doc in docs:
            doc_str = str(doc)
            if doc_str not in corpus_strs:
                raise ValueError(f"Document '{doc_str}' not found in corpus.")

            # Get the first occurrence index for labeling
            idx = corpus_strs.index(doc_str)
            doc_labels.append(self.labels[idx])

        return doc_labels

    def _create_doc_mapping(self) -> dict[str, list[int]]:
        """Create mapping from document content to corpus indices."""
        doc_to_indices = {}
        for i, corpus_doc in enumerate(self.corpus):
            doc_str = str(corpus_doc)
            if doc_str not in doc_to_indices:
                doc_to_indices[doc_str] = []
            doc_to_indices[doc_str].append(i)
        return doc_to_indices

    def _determine_background_categories(
        self, cat: str, background_cats: Optional[list[str]], available_cats: set[str]
    ) -> list[str]:
        """Determine which categories to use as background."""
        if background_cats is None:
            background_categories = [c for c in available_cats if c != cat]
        else:
            # Validate provided background categories
            invalid_cats = [c for c in background_cats if c not in available_cats]
            if invalid_cats:
                sorted_cats = sorted(available_cats)
                raise ValueError(
                    f"Invalid background categories: {invalid_cats}. "
                    f"Available categories: {sorted_cats}"
                )

            if cat in background_cats:
                raise ValueError(
                    f"Target category '{cat}' cannot be included in background_cats."
                )

            background_categories = list(background_cats)

        if not background_categories:
            raise ValueError(
                f"No valid background categories available. "
                f"Target category '{cat}' is the only category in the corpus."
            )

        return background_categories

    def _separate_docs_by_category(
        self, target_cat: str, background_categories: list[str]
    ) -> tuple[list[Doc | str], list[Doc | str]]:
        """Separate documents into target and background groups by category."""
        target_documents = []
        background_documents = []

        if self.cats is None:
            raise ValueError(
                "Categories (cats) must be provided for separating documents by category."
            )
        for doc, doc_cat in zip(self.corpus, self.cats):
            if doc_cat == target_cat:
                target_documents.append(doc)
            elif doc_cat in background_categories:
                background_documents.append(doc)

        # Validate we have documents in both groups
        if not target_documents:
            raise ValueError(f"No documents found for target category '{target_cat}'.")

        if not background_documents:
            cats_str = "', '".join(background_categories)
            raise ValueError(
                f"No documents found in background categories ['{cats_str}']. "
                f"Ensure these categories contain documents."
            )

        return target_documents, background_documents

    def _format_output(self, results):
        """Format the output according to the output_format"""
        if self.output_format == "dict":
            return results
        elif self.output_format == "dataframe":
            return self.to_df(results)
        elif self.output_format == "list_of_dicts":
            return self.to_list_of_dicts(results)
        elif self.output_format == "list_of_tuples":
            return self.to_list_of_tuples(results)
        else:
            return results  # fallback to raw results

    @staticmethod
    def to_df(results):
        """Return results as a pandas DataFrame with columns: label, term, z_score."""
        import pandas as pd

        rows = []

        def extract_rows(label, ztest_result):
            # ztest_result can be a ZTest instance or a dict with 'result'
            if hasattr(ztest_result, "topwords"):
                topwords = getattr(ztest_result, "topwords", []) or []
            elif isinstance(ztest_result, dict) and "result" in ztest_result:
                return extract_rows(label, ztest_result["result"])
            else:
                topwords = []
            for term, z_score in topwords:
                rows.append({"label": label, "term": term, "z_score": z_score})

        if isinstance(results, list):
            for item in results:
                label = item.get("label", None)
                result = item.get("result", None)
                if label is not None and result is not None:
                    extract_rows(label, result)
        elif isinstance(results, dict):
            # Try to handle dicts of the form {cat: {category, background_categories, result}}
            for group, comparison_result in results.items():
                # If nested dict with 'result', use group as label
                if (
                    isinstance(comparison_result, dict)
                    and "result" in comparison_result
                ):
                    extract_rows(group, comparison_result["result"])
                # If dict with 'label' and 'result'
                elif (
                    isinstance(comparison_result, dict)
                    and "label" in comparison_result
                    and "result" in comparison_result
                ):
                    extract_rows(
                        comparison_result["label"], comparison_result["result"]
                    )
                # If already a ZTest instance
                elif hasattr(comparison_result, "topwords"):
                    extract_rows(group, comparison_result)
        # Return DataFrame
        return pd.DataFrame(rows, columns=["label", "term", "z_score"])

    @staticmethod
    def to_list_of_dicts(results):
        """Return results as a list of dicts with keys: label, term, z_score."""
        output = []

        def extract(label, ztest_result):
            if hasattr(ztest_result, "topwords"):
                # Ensure topwords is always a list, even if None
                topwords = getattr(ztest_result, "topwords", []) or []
            elif isinstance(ztest_result, dict) and "result" in ztest_result:
                return extract(label, ztest_result["result"])
            else:
                topwords = []
            for term, z_score in topwords:
                output.append({"label": label, "term": term, "z_score": z_score})

        if isinstance(results, list):
            for item in results:
                label = item.get("label", None)
                result = item.get("result", None)
                if label is not None and result is not None:
                    extract(label, result)
        elif isinstance(results, dict):
            for group, comparison_result in results.items():
                if (
                    isinstance(comparison_result, dict)
                    and "result" in comparison_result
                ):
                    extract(group, comparison_result["result"])
                elif (
                    isinstance(comparison_result, dict)
                    and "label" in comparison_result
                    and "result" in comparison_result
                ):
                    extract(comparison_result["label"], comparison_result["result"])
                elif hasattr(comparison_result, "topwords"):
                    extract(group, comparison_result)
        return output

    @staticmethod
    def to_list_of_tuples(results):
        """Return results as a list of (label, term, z_score) tuples."""
        output = []

        def extract(label, ztest_result):
            if hasattr(ztest_result, "topwords"):
                # Ensure topwords is always a list, even if None
                topwords = getattr(ztest_result, "topwords", []) or []
            elif isinstance(ztest_result, dict) and "result" in ztest_result:
                return extract(label, ztest_result["result"])
            else:
                topwords = []
            for term, z_score in topwords:
                output.append((label, term, z_score))

        if isinstance(results, list):
            for item in results:
                label = item.get("label", None)
                result = item.get("result", None)
                if label is not None and result is not None:
                    extract(label, result)
        elif isinstance(results, dict):
            for group, comparison_result in results.items():
                if (
                    isinstance(comparison_result, dict)
                    and "result" in comparison_result
                ):
                    extract(group, comparison_result["result"])
                elif (
                    isinstance(comparison_result, dict)
                    and "label" in comparison_result
                    and "result" in comparison_result
                ):
                    extract(comparison_result["label"], comparison_result["result"])
                elif hasattr(comparison_result, "topwords"):
                    extract(group, comparison_result)
        return output
