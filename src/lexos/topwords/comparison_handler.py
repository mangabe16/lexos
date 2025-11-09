"""comparison_handler.py.

Last Update: November 8, 2025
Last Tested: TBD

# --- ADDITION FOR ALL COMPARISON METHODS SUPPORT ---
# Can be used in both keyterms.py and ztest.py
"""

from collections import defaultdict
from typing import Any, Optional

import pandas as pd


class ComparisonHandler:
    """Comparison handler which does all 3 comparison types as shown in the lexos web app."""

    def __init__(
        self,
        cls,
        labels: Optional[list[str]] = None,
        doc_content_to_label_map: Optional[
            dict[str, str]
        ] = None,  # Added new parameter
        output_format: str = "dict",
        **kwargs,
    ):
        """Initialize the ComparisonHandler with a comparison class, output format, and optional keyword arguments.

        Args:
            cls (Any): The comparison class (e.g., ZTest) to be used for comparisons.
            labels (list[str], optional): A list of labels corresponding to the documents for `compare_each_doc_to_corpus`.
                If not provided, documents will be numbered (e.g., "Doc 1", "Doc 2").
            doc_content_to_label_map (dict[str, str], optional): A dictionary mapping document content (as a string) to its desired label.
                This is used by `compare_each_doc_to_other_classes` to provide
                more descriptive labels than just "ClassName Doc X".
            **kwargs (dict, optional): Additional keyword arguments to pass to the comparison class constructor.
        """
        self.cls = cls
        self.labels = labels
        self.doc_content_to_label_map = doc_content_to_label_map  # Store the new map
        self.kwargs = kwargs
        self.output_format = output_format

    def compare_each_doc_to_corpus(self, documents: list[str]) -> list[dict]:
        """Compare each document to the rest of the corpus (all other documents).

        Args:
            documents (list[str]): List of documents to compare.

        Returns:
            list[dict]: List of dictionaries containing comparison results.
            List of dictionaries, each containing a 'label' and the comparison 'result'
            for each document compared to the rest of the corpus.
        """
        # Validate that if labels are provided, their count matches the documents
        if self.labels is not None and len(self.labels) != len(documents):
            raise ValueError(
                "If labels are provided for 'compare_each_doc_to_corpus', "
                "their count must match the document count."
            )

        results = []
        for i, doc in enumerate(documents):
            # Create the background corpus by excluding the current document
            background = documents[:i] + documents[i + 1 :]

            # Initialize the comparison class instance
            instance = self.cls(
                target_documents=[doc], background_documents=background, **self.kwargs
            )
            # Use label if provided, else fallback
            label = (
                self.labels[i]
                if self.labels is not None and i < len(self.labels)
                else f"Doc {i + 1}"
            )
            result = instance()
            # Attach label to result dict
            if isinstance(result, dict):
                result = {"label": label, **result}
            else:
                result = {"label": label, "result": result}
            results.append(result)
        return self._format_output(results)

    def compare_each_doc_to_other_classes(
        self, class_docs: dict[str, list[str]]
    ) -> dict[str, list[dict]]:
        """Compare each document in each class to all documents in other classes.

        Args:
            class_docs (dict[str, list[str]]): Dictionary mapping class names to lists of documents.

        Returns:
            dict[str, list[dict]]: Dictionary mapping class names to lists of dictionaries.
            Dictionary mapping class names to lists of dictionaries. Each inner
            dictionary contains a 'label' (derived from class name and document number)
            and the comparison 'result' for each document compared to all documents in other classes.
            Note: The 'labels' parameter passed to __init__ is primarily for `compare_each_doc_to_corpus`
            and is not directly applied here, as class names are inherent labels.
        """
        results = defaultdict(list)
        for cls_name, docs in class_docs.items():
            # Build the background corpus from all documents in other classes
            background = [
                d
                for other_cls, other_docs in class_docs.items()
                if other_cls != cls_name
                for d in other_docs
            ]
            for i, doc in enumerate(docs):
                # Initialize the comparison class instance for the current document
                instance = self.cls(
                    target_documents=[doc],
                    background_documents=background,
                    **self.kwargs,
                )
                # Determine the label for the current document
                # Use provided label from the map, otherwise fall back to class name and numbering
                doc_label = (
                    self.doc_content_to_label_map.get(doc, f"{cls_name} Doc {i + 1}")
                    if self.doc_content_to_label_map
                    else f"{cls_name} Doc {i + 1}"
                )
                results[cls_name].append({"label": doc_label, "result": instance()})
        return self._format_output(dict(results))

    def compare_each_class_to_other_classes(
        self, class_docs: dict[str, list[str]]
    ) -> dict[str, dict]:
        """Compare each class (group of documents) to all documents in other classes.

        Args:
            class_docs (dict[str, list[str]]): Dictionary mapping class names to lists of documents.

        Returns:
            dict[str, dict]: Dictionary mapping class names to results for each class compared to all documents in other classes.
                The class names themselves serve as the labels for these comparisons.
        """
        results = {}
        for cls_name, docs in class_docs.items():
            # Build the background corpus from all documents in other classes
            background = [
                d
                for other_cls, other_docs in class_docs.items()
                if other_cls != cls_name
                for d in other_docs
            ]
            # Initialize the comparison class instance for the entire class
            instance = self.cls(
                target_documents=docs, background_documents=background, **self.kwargs
            )
            # The class name is the natural label for this comparison type
            results[cls_name] = instance()
        return self._format_output(results)

    def _format_output(self, results: Any) -> Any:
        """Format the output according to the output_format.

        Args:
            results (Any): The raw results from the comparison methods.

        Returns:
            Any: The formatted results based on the specified output format.
        """
        if self.output_format == "dict":
            return results
        elif self.output_format == "dataframe":
            return self.to_df(results)
        elif self.output_format == "list_of_dicts":
            return self.to_list_of_dicts(results)
        else:
            return results  # fallback to raw results

    @staticmethod
    def to_df(results: Any) -> pd.DataFrame:
        """Return results as a pandas DataFrame, flattened so each row is a single topword.

        Args:
            results (Any): The raw results from the comparison methods.

        Returns:
            pd.DataFrame: The results formatted as a pandas DataFrame.
        """
        rows = []
        # Handle list of dicts (e.g., from compare_each_doc_to_corpus)
        if isinstance(results, list):
            for i, res in enumerate(results):
                label = (
                    res.get("label", f"Doc {i + 1}")
                    if isinstance(res, dict)
                    else f"Doc {i + 1}"
                )
                # Support both {'topwords': [...]} and just a list of dicts
                topwords = res.get("topwords", []) if isinstance(res, dict) else []
                for tw in topwords:
                    row = {"label": label}
                    row.update(tw)
                    rows.append(row)
        # Handle dict of dicts (e.g., from compare_each_class_to_other_classes)
        elif isinstance(results, dict):
            # Check if values are dicts (not lists)
            if all(isinstance(v, dict) for v in results.values()):
                for group, res in results.items():
                    label = group
                    topwords = res.get("topwords", []) if isinstance(res, dict) else []
                    for tw in topwords:
                        row = {"group": group, "label": label}
                        row.update(tw)
                        rows.append(row)
            else:
                # existing code for dict of lists
                for group, group_results in results.items():
                    for res in group_results:
                        label = (
                            res.get("label", group) if isinstance(res, dict) else group
                        )
                        topwords = (
                            res.get("result", {}).get("topwords", [])
                            if isinstance(res, dict)
                            else []
                        )
                        for tw in topwords:
                            row = {"group": group, "label": label}
                            row.update(tw)
                            rows.append(row)
        else:
            raise ValueError("Unsupported results format for DataFrame conversion.")
        return pd.DataFrame(rows)

    @staticmethod
    def to_list_of_dicts(results: Any) -> list[dict]:
        """Return results as a list of dicts.

        Args:
            results (Any): The raw results from the comparison methods.

        Returns:
            list[dict]: The results formatted as a list of dictionaries.
        """
        if isinstance(results, list):
            return results
        elif isinstance(results, dict):
            all_dicts = []
            for group, group_results in results.items():
                for res in group_results:
                    if isinstance(res, dict):
                        d = {"group": group}
                        d.update(res)
                        all_dicts.append(d)
                    elif isinstance(res, (list, tuple)) and len(res) == 2:
                        all_dicts.append(
                            {"group": group, "term": res[0], "score": res[1]}
                        )
            return all_dicts
        else:
            raise ValueError("Unsupported results format for list_of_dicts conversion.")
