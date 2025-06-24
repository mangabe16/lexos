# --- ADDITION FOR ALL COMPARISON METHODS SUPPORT ---
# Can be used in both keyterms.py and ztest.py

from collections import defaultdict


class ComparisonHandler:
    """Comparison handler which does all 3 comparison types as shown in the lexos web app."""

    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.kwargs = kwargs

    def compare_each_doc_to_corpus(self, documents: list[str]) -> list[dict]:
        results = []
        for i, doc in enumerate(documents):
            background = documents[:i] + documents[i + 1 :]  # all except current
            instance = self.cls(
                target_documents=[doc], background_documents=background, **self.kwargs
            )
            results.append(instance())
        return results

    def compare_each_doc_to_other_classes(
        self, class_docs: dict[str, list[str]]
    ) -> dict[str, list[dict]]:
        results = defaultdict(list)
        for cls_name, docs in class_docs.items():
            background = [
                d
                for other_cls, other_docs in class_docs.items()
                if other_cls != cls_name
                for d in other_docs
            ]
            for doc in docs:
                instance = self.cls(
                    target_documents=[doc],
                    background_documents=background,
                    **self.kwargs,
                )
                results[cls_name].append(instance())
        return results

    def compare_each_class_to_other_classes(
        self, class_docs: dict[str, list[str]]
    ) -> dict[str, dict]:
        results = {}
        for cls_name, docs in class_docs.items():
            background = [
                d
                for other_cls, other_docs in class_docs.items()
                if other_cls != cls_name
                for d in other_docs
            ]
            instance = self.cls(
                target_documents=docs, background_documents=background, **self.kwargs
            )
            results[cls_name] = instance()
        return results
