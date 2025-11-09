"""compare.py.

Last Update: November 8, 2025
Last Tested: November 8, 2025

Provides a unified interface for three types of document comparison operations:

1. compare_each_doc_to_corpus(documents)

**Compares each individual document to all other documents in the corpus.**

- For each document, creates a background corpus from all the other documents
- Useful for finding what makes each document unique compared to the rest
- Returns a list of results, one per document

Example use case: "What terms distinguish Document A from all other documents?"

2. compare_each_doc_to_other_classes(class_docs)

**Compares each document within a class to all documents in other classes.**

- Takes a dictionary mapping class names to lists of documents <<Should be able to build the dict if the docs have custom extensions>>
- For each document in a class, compares it to all documents in all other classes
- Returns results grouped by class

Example use case: "In a corpus of Shakespeare, Marlowe, and Jonson plays, what terms distinguish each individual Shakespeare play from all Marlowe and Jonson plays?"

3. compare_each_class_to_other_classes(class_docs)

**Compares entire classes (groups of documents) to all other classes.**

- Treats all documents in a class as a single unit
- Compares the entire class to all documents in other classes
- Returns one result per class

Example use case: "What terms distinguish all Shakespeare plays collectively from all Marlowe and Jonson plays?"

Usage:

# Create configured instance with all desired parameters
ztest = ZTest(target_docs=[], comparison_docs=[], topn=20, ngrams=2, case_sensitive=False)

# Pass instance to Comparison
comparison = Comparison(comparison_instance=ztest)

# Use it
results = comparison.compare_each_doc_to_corpus(documents)

compare_each_doc_to_corpus accepts lists of strings of spaCy Docs. The other two methods accept either a class_documents dict or a list of spaCy Docs with class_names. Examples below:

Usage Option 1 with class_documents dict
# With strings
class_docs = {
    "Shakespeare": ["Hamlet text", "Macbeth text"],
    "Marlowe": ["Faustus text", "Jew of Malta text"]
}
comparison.compare_each_doc_to_other_classes(class_documents=class_docs)

# With spaCy Docs (no extension required)
class_docs = {
    "Shakespeare": [nlp("Hamlet"), nlp("Macbeth")],
    "Marlowe": [nlp("Faustus"), nlp("Jew of Malta")]
}
comparison.compare_each_class_to_other_classes(class_documents=class_docs)

Usage Option 2 with list of spaCy Docs and class_names

# Use the docs + class_names parameters
comparison.compare_each_doc_to_other_classes(
    docs=docs,
    class_names=["author"]  # Check the "author" extension
)

# Can specify multiple extensions (uses first one found)
comparison.compare_each_class_to_other_classes(
    docs=docs,
    class_names=["author", "category", "genre"]
)
"""

from collections import defaultdict
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc

from lexos.exceptions import LexosException
from lexos.topwords import TopWords

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
)


class Comparison(BaseModel):
    """Handler for performing document comparison operations with flexible output formats."""

    comparison_instance: TopWords = Field(
        ...,
        description="A configured instance of a comparison class (e.g., ZTest, KeyTerms).",
    )
    document_labels: Optional[list[str]] = Field(
        default=None, description="Labels for the documents being compared."
    )
    document_to_label_map: Optional[dict[str, str]] = Field(
        default_factory=dict, description="Mapping of document content to labels."
    )
    output_format: str = Field(
        default="dict", description="The output format for the comparison results."
    )

    model_config = validation_config

    def __init__(self, **data) -> None:
        """Initialize the Comparison class."""
        super().__init__(**data)

    def compare_each_doc_to_corpus(
        self, documents: list[str] | list[Doc], output_format: Optional[str] = None
    ) -> Any:
        """Compare each document to all other documents in the corpus.

        For each document, creates a comparison corpus from all other documents
        and performs the comparison.

        Args:
            documents: List of document texts (strings) or spaCy Doc objects to compare.
            output_format: Optional output format override. If None, uses self.output_format.

        Returns:
            Comparison results in the specified output format.

        Raises:
            ValueError: If document_labels count doesn't match documents count.
        """
        # Convert spaCy Docs to text strings
        text_documents = self._extract_text_from_documents(documents)

        self._validate_document_labels(len(text_documents))

        comparison_results = []
        for doc_index, target_doc in enumerate(text_documents):
            # Build comparison corpus excluding current document
            comparison_corpus = self._build_comparison_corpus(text_documents, doc_index)

            # Get label for this document
            doc_label = self._get_document_label(doc_index)

            # Perform comparison
            result = self._run_comparison([target_doc], comparison_corpus)
            comparison_results.append({"label": doc_label, **result})

        return self._format_output(comparison_results, output_format)

    def compare_each_doc_to_other_classes(
        self,
        class_documents: Optional[dict[str, list[str] | list[Doc]]] = None,
        docs: Optional[list[Doc]] = None,
        class_names: Optional[list[str]] = None,
        output_format: Optional[str] = None,
    ) -> Any:
        """Compare each document in a class to all documents in other classes.

        Args:
            class_documents: Dictionary mapping class names to lists of documents.
                If provided, this takes precedence over docs and class_names.
            docs: List of spaCy Doc objects. Requires class_names parameter.
            class_names: List of custom extension names to use for extracting
                class labels from Doc objects. Required when docs is provided.
            output_format: Optional output format override. If None, uses self.output_format.

        Returns:
            Comparison results in the specified output format.

        Raises:
            LexosException: If invalid parameter combinations are provided.
        """
        # Convert to dict format if docs provided
        class_documents_dict = self._prepare_class_documents(
            class_documents=class_documents, docs=docs, class_names=class_names
        )

        # Convert all documents to text strings
        text_class_documents = self._extract_text_from_class_documents(
            class_documents_dict
        )

        results_by_class = defaultdict(list)

        for class_name, documents_in_class in text_class_documents.items():
            # Build comparison from all other classes
            comparison_corpus = self._build_other_classes_comparison(
                text_class_documents, class_name
            )

            # Compare each document in this class to the comparison
            for doc_index, target_doc in enumerate(documents_in_class):
                doc_label = self._get_class_document_label(
                    target_doc, class_name, doc_index
                )
                result = self._run_comparison([target_doc], comparison_corpus)
                results_by_class[class_name].append(
                    {"label": doc_label, "result": result}
                )

        return self._format_output(dict(results_by_class), output_format)

    def compare_each_class_to_other_classes(
        self,
        class_documents: Optional[dict[str, list[str] | list[Doc]]] = None,
        docs: Optional[list[Doc]] = None,
        class_names: Optional[list[str]] = None,
        output_format: Optional[str] = None,
    ) -> Any:
        """Compare each class (all its documents) to all documents in other classes.

        Args:
            class_documents: Dictionary mapping class names to lists of documents.
                If provided, this takes precedence over docs and class_names.
            docs: List of spaCy Doc objects. Requires class_names parameter.
            class_names: List of custom extension names to use for extracting
                class labels from Doc objects. Required when docs is provided.
            output_format: Optional output format override. If None, uses self.output_format.

        Returns:
            Comparison results in the specified output format.

        Raises:
            LexosException: If invalid parameter combinations are provided.
        """
        # Convert to dict format if docs provided
        class_documents_dict = self._prepare_class_documents(
            class_documents=class_documents, docs=docs, class_names=class_names
        )

        # Convert all documents to text strings
        text_class_documents = self._extract_text_from_class_documents(
            class_documents_dict
        )

        results_by_class = {}

        for class_name, documents_in_class in text_class_documents.items():
            # Build comparison from all other classes
            comparison_corpus = self._build_other_classes_comparison(
                text_class_documents, class_name
            )

            # Compare entire class to the comparison
            result = self._run_comparison(documents_in_class, comparison_corpus)
            results_by_class[class_name] = result

        return self._format_output(results_by_class, output_format)

    # ---- Private helper methods ----

    def _prepare_class_documents(
        self,
        class_documents: Optional[dict[str, list[str] | list[Doc]]] = None,
        docs: Optional[list[Doc]] = None,
        class_names: Optional[list[str]] = None,
    ) -> dict[str, list[str] | list[Doc]]:
        """Prepare class documents, converting docs list to dict if needed.

        Args:
            class_documents: Optional dict mapping class names to document lists.
            docs: Optional list of Doc objects.
            class_names: Optional list of custom extension names to check for class labels.

        Returns:
            Dictionary mapping class names to lists of documents.

        Raises:
            LexosException: If invalid parameter combinations are provided.
        """
        # Case 1: class_documents provided - use it directly (overrides everything else)
        if class_documents is not None:
            if docs is not None or class_names is not None:
                raise LexosException(
                    "Cannot provide both 'class_documents' and 'docs'/'class_names'. "
                    "Use 'class_documents' alone, or use 'docs' with 'class_names'."
                )
            return class_documents

        # Case 2: docs provided - must have class_names
        if docs is not None:
            if not docs:
                raise LexosException(
                    "Empty list of documents provided in 'docs' parameter."
                )

            # Verify all items are Doc objects
            if not all(isinstance(doc, Doc) for doc in docs):
                raise LexosException(
                    "All items in 'docs' parameter must be spaCy Doc objects."
                )

            # class_names is required when using docs
            if class_names is None or not class_names:
                raise LexosException(
                    "When 'docs' is provided, you must also provide 'class_names' "
                    "to specify which custom extensions to use for class labels."
                )

            # Build the dict from Doc extensions
            return self._build_class_dict_from_extensions(docs, class_names)

        # Case 3: Neither provided - error
        raise LexosException(
            "You must provide either 'class_documents' (a dict) or both 'docs' and 'class_names'."
        )

    def _build_class_dict_from_extensions(
        self, docs: list[Doc], class_names: list[str]
    ) -> dict[str, list[Doc]]:
        """Build a class document dictionary from Doc custom extensions.

        Args:
            docs: List of spaCy Doc objects.
            class_names: List of custom extension names to check.

        Returns:
            Dictionary mapping class labels to lists of Doc objects.

        Raises:
            LexosException: If extension attributes are not found or invalid.
        """
        class_dict = defaultdict(list)

        for doc_index, doc in enumerate(docs):
            class_label = None

            # Try each extension name until we find one
            for ext_name in class_names:
                if not ext_name.startswith("_"):
                    ext_name = f"_.{ext_name}"

                # Check if the extension exists
                if not doc.has_extension(ext_name.lstrip("_.")):
                    continue

                # Get the extension value
                try:
                    parts = ext_name.split(".")
                    value = doc
                    for part in parts:
                        if part == "_":
                            value = value._
                        else:
                            value = getattr(value, part)

                    if value is not None:
                        class_label = str(value)
                        break
                except AttributeError:
                    continue

            # If no valid extension found, raise error
            if class_label is None:
                available_extensions = [
                    name for name in dir(doc._) if not name.startswith("_")
                ]
                raise LexosException(
                    f"Document at index {doc_index} does not have any of the specified "
                    f"custom extensions: {class_names}. "
                    f"Available extensions: {available_extensions if available_extensions else 'None'}"
                )

            class_dict[class_label].append(doc)

        if not class_dict:
            raise LexosException(
                "Could not extract any class labels from the documents using the "
                f"provided extension names: {class_names}"
            )

        return dict(class_dict)

    def _extract_text_from_documents(
        self, documents: list[str] | list[Doc]
    ) -> list[str]:
        """Extract text from documents, handling both strings and spaCy Doc objects.

        Args:
            documents: List of strings or spaCy Doc objects.

        Returns:
            List of text strings.
        """
        text_documents = []
        for doc in documents:
            if isinstance(doc, Doc):
                text_documents.append(doc.text)
            elif isinstance(doc, str):
                text_documents.append(doc)
            else:
                raise LexosException(
                    f"Unsupported document type: {type(doc)}. "
                    f"Expected str or spacy.tokens.Doc."
                )
        return text_documents

    def _extract_text_from_class_documents(
        self, class_documents: dict[str, list[str] | list[Doc]]
    ) -> dict[str, list[str]]:
        """Extract text from class documents, handling both strings and spaCy Doc objects.

        Args:
            class_documents: Dictionary mapping class names to lists of strings or spaCy Docs.

        Returns:
            Dictionary mapping class names to lists of text strings.
        """
        text_class_documents = {}
        for class_name, documents in class_documents.items():
            text_class_documents[class_name] = self._extract_text_from_documents(
                documents
            )
        return text_class_documents

    def _validate_document_labels(self, document_count: int) -> None:
        """Validate that document labels count matches document count."""
        if (
            self.document_labels is not None
            and len(self.document_labels) != document_count
        ):
            raise LexosException(
                f"Document labels count ({len(self.document_labels)}) must match "
                f"document count ({document_count})."
            )

    def _build_comparison_corpus(
        self, all_documents: list[str], exclude_index: int
    ) -> list[str]:
        """Build a comparison corpus excluding the document at exclude_index."""
        return all_documents[:exclude_index] + all_documents[exclude_index + 1 :]

    def _build_other_classes_comparison(
        self, class_documents: dict[str, list[str]], current_class: str
    ) -> list[str]:
        """Build a comparison corpus from all documents in classes other than current_class."""
        comparison = []
        for other_class_name, other_class_docs in class_documents.items():
            if other_class_name != current_class:
                comparison.extend(other_class_docs)
        return comparison

    def _get_document_label(self, doc_index: int) -> str:
        """Get the label for a document at the given index."""
        if self.document_labels and doc_index < len(self.document_labels):
            return self.document_labels[doc_index]
        return f"Doc {doc_index + 1}"

    def _get_class_document_label(
        self, document: str, class_name: str, doc_index: int
    ) -> str:
        """Get the label for a document within a class."""
        if document in self.document_to_label_map:
            return self.document_to_label_map[document]
        return f"{class_name} Doc {doc_index + 1}"

    def _run_comparison(
        self, target_documents: list[str], comparison_documents: list[str]
    ) -> dict:
        """Run the comparison using the configured comparison instance.

        Creates a new instance of the same type with the same configuration,
        but with the specified target and comparison documents.

        Args:
            target_documents: The target documents to compare.
            comparison_documents: The comparison/background documents.

        Returns:
            Dictionary containing the comparison results.
        """
        # Get the class type from the instance
        comparison_class = type(self.comparison_instance)

        # Get the model dump (all field values) from the template instance
        # Exclude the target_docs and comparison_docs fields
        template_config = self.comparison_instance.model_dump(
            exclude={"target_docs", "comparison_docs"}
        )

        # Create a new instance with the target/comparison docs and template config
        comparison_instance = comparison_class(
            target_docs=target_documents,
            comparison_docs=comparison_documents,
            **template_config,
        )

        result = comparison_instance.to_dict()

        # Ensure result is a dictionary
        if not isinstance(result, dict):
            result = {"result": result}

        return result

    def _format_output(self, results: Any, output_format: Optional[str] = None) -> Any:
        """Format the output according to the configured output format.

        Args:
            results: The raw comparison results.
            output_format: Optional format override. If None, uses self.output_format.

        Returns:
            Formatted results.
        """
        # Use the override if provided, otherwise use the class attribute
        format_to_use = (
            output_format if output_format is not None else self.output_format
        )

        format_handlers = {
            "dict": lambda r: r,
            "dataframe": self._to_dataframe,
            "list_of_dicts": self._to_list_of_dicts,
        }

        handler = format_handlers.get(format_to_use)
        if handler is None:
            return results  # Fallback to raw results

        return handler(results)

    def _to_dataframe(self, results: Any) -> pd.DataFrame:
        """Convert results to a flattened pandas DataFrame.

        Each row represents a single term/topword with its associated metadata.

        Args:
            results: Raw comparison results.

        Returns:
            DataFrame with columns for labels, groups, and term-level data.
        """
        rows = []

        if isinstance(results, list):
            rows = self._extract_rows_from_list(results)
        elif isinstance(results, dict):
            rows = self._extract_rows_from_dict(results)
        else:
            raise LexosException(
                f"Unsupported results type for DataFrame conversion: {type(results)}"
            )

        return pd.DataFrame(rows)

    def _extract_rows_from_list(self, results: list) -> list[dict]:
        """Extract rows from list-based results (e.g., compare_each_doc_to_corpus)."""
        rows = []
        for doc_index, doc_result in enumerate(results):
            doc_label = self._extract_label(doc_result, f"Doc {doc_index + 1}")
            topwords = self._extract_topwords(doc_result)

            for term_data in topwords:
                row = {"label": doc_label, **term_data}
                rows.append(row)

        return rows

    def _extract_rows_from_dict(self, results: dict) -> list[dict]:
        """Extract rows from dict-based results (e.g., class comparisons)."""
        # Check if values are dicts (compare_each_class_to_other_classes)
        if all(isinstance(v, dict) for v in results.values()):
            return self._extract_rows_from_class_results(results)

        # Otherwise, handle dict of lists (compare_each_doc_to_other_classes)
        return self._extract_rows_from_grouped_results(results)

    def _extract_rows_from_class_results(self, results: dict) -> list[dict]:
        """Extract rows from class-level comparison results."""
        rows = []
        for class_name, class_result in results.items():
            topwords = self._extract_topwords(class_result)

            for term_data in topwords:
                row = {"group": class_name, "label": class_name, **term_data}
                rows.append(row)

        return rows

    def _extract_rows_from_grouped_results(self, results: dict) -> list[dict]:
        """Extract rows from grouped document comparison results."""
        rows = []
        for group_name, group_results in results.items():
            for doc_result in group_results:
                doc_label = self._extract_label(doc_result, group_name)

                # Handle nested result structure
                result_data = (
                    doc_result.get("result", {}) if isinstance(doc_result, dict) else {}
                )
                topwords = self._extract_topwords(result_data)

                for term_data in topwords:
                    row = {"group": group_name, "label": doc_label, **term_data}
                    rows.append(row)

        return rows

    def _extract_label(self, result_dict: dict, default_label: str) -> str:
        """Extract the label from a result dictionary."""
        if isinstance(result_dict, dict):
            return result_dict.get("label", default_label)
        return default_label

    def _extract_topwords(self, result_dict: Any) -> list[dict]:
        """Extract the topwords list from a result dictionary."""
        if isinstance(result_dict, dict):
            return result_dict.get("topwords", [])
        return []

    def _to_list_of_dicts(self, results: Any) -> list[dict]:
        """Convert results to a list of dictionaries.

        Args:
            results: Raw comparison results.

        Returns:
            List of dictionaries with flattened structure.
        """
        if isinstance(results, list):
            return results

        if isinstance(results, dict):
            return self._flatten_dict_results(results)

        raise LexosException(
            f"Unsupported results type for list_of_dicts conversion: {type(results)}"
        )

    def _flatten_dict_results(self, results: dict) -> list[dict]:
        """Flatten dictionary results into a list of dictionaries."""
        flattened = []

        for group_name, group_results in results.items():
            if not isinstance(group_results, list):
                continue

            for doc_result in group_results:
                if isinstance(doc_result, dict):
                    flattened_entry = {"group": group_name, **doc_result}
                    flattened.append(flattened_entry)
                elif isinstance(doc_result, (list, tuple)) and len(doc_result) == 2:
                    flattened.append(
                        {
                            "group": group_name,
                            "term": doc_result[0],
                            "score": doc_result[1],
                        }
                    )

        return flattened
