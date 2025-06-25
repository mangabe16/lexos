from collections import defaultdict
from typing import Optional, Any


class ComparisonHandler:
    """Comparison handler which does all 3 comparison types as shown in the lexos web app."""

    def __init__(
        self,
        cls,
        labels: Optional[list[str]] = None,
        doc_content_to_label_map: Optional[
            dict[str, str]
        ] = None,  # Added new parameter
        **kwargs,
    ):
        """Initialize the ComparisonHandler with a comparison class and optional keyword arguments.

        Parameters
        ----------
        cls : Any
            The comparison class (e.g., ZTest) to be used for comparisons.
        labels : list of str, optional
            A list of labels corresponding to the documents for `compare_each_doc_to_corpus`.
            If not provided, documents will be numbered (e.g., "Doc 1", "Doc 2").
        doc_content_to_label_map : dict of str to str, optional
            A dictionary mapping document content (as a string) to its desired label.
            This is used by `compare_each_doc_to_other_classes` to provide
            more descriptive labels than just "ClassName Doc X".
        **kwargs : dict
            Additional keyword arguments to pass to the comparison class constructor.
        """
        self.cls = cls
        self.labels = labels
        self.doc_content_to_label_map = doc_content_to_label_map  # Store the new map
        self.kwargs = kwargs

    def compare_each_doc_to_corpus(self, documents: list[str]) -> list[dict]:
        """Compare each document to the rest of the corpus (all other documents).

        Parameters
        ----------
        documents : list of str
            List of documents to compare.

        Returns:
        -------
        list of dict
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

            # Determine the label for the current document
            # Use provided label if available, otherwise fall back to numbering
            doc_label = self.labels[i] if self.labels else f"Doc {i + 1}"

            # Append the result with its label
            results.append({"label": doc_label, "result": instance()})
        return results

    def compare_each_doc_to_other_classes(
        self, class_docs: dict[str, list[str]]
    ) -> dict[str, list[dict]]:
        """Compare each document in each class to all documents in other classes.

        Parameters
        ----------
        class_docs : dict of str to list of str
            Dictionary mapping class names to lists of documents.

        Returns:
        -------
        dict of str to list of dict
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
        return results

    def compare_each_class_to_other_classes(
        self, class_docs: dict[str, list[str]]
    ) -> dict[str, dict]:
        """Compare each class (group of documents) to all documents in other classes.

        Parameters
        ----------
        class_docs : dict of str to list of str
            Dictionary mapping class names to lists of documents.

        Returns:
        -------
        dict of str to dict
            Dictionary mapping class names to results for each class compared to all documents in other classes.
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
        return results
