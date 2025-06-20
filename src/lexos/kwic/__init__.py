"""__init__.py.

Last Updated: 6/12/25
Last Tested: 6/18/25

Current Usage:
- Find keywords within their context in a spaCy doc or string
"""

from textacy.extract import kwic
from spacy.tokens import Doc
from typing import Iterable, Pattern
from pydantic import validate_call, ConfigDict
import pandas as pd

from lexos.exceptions import LexosException


class Kwic:
    """A class for generating keyword-in-context (KWIC) results using textacy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validate_call(config=model_config)
    def find(
        doc: Doc | str,
        keyword: str | Pattern,
        ignore_case: bool = True,
        window_size: int = 50,
        pad_context: bool = False,
    ) -> Iterable[tuple[str, str, str]]:
        """Generate KWIC results for a given term in the document."""
        return kwic.keyword_in_context(
            doc=doc,
            keyword=keyword,
            ignore_case=ignore_case,
            window_width=window_size,
            pad_context=pad_context,
        )

    @validate_call(config=model_config)
    def find_to_dataframe(
        doc: Doc | str,
        keyword: str | Pattern,
        ignore_case: bool = True,
        window_size: int = 50,
        pad_context: bool = False,
    ) -> pd.DataFrame:
        """Generate KWIC results for a given term in the document in the form of a pandas DataFrame."""
        return pd.DataFrame(
            kwic.keyword_in_context(
                doc=doc,
                keyword=keyword,
                ignore_case=ignore_case,
                window_width=window_size,
                pad_context=pad_context,
            ),
            columns=["Left", "Keyword", "Right"],
        )

    @validate_call(config=model_config)
    def find_multiple_keywords(
        doc: Doc | str,
        keywords: Iterable[str | Pattern],
        ignore_case: bool = True,
        window_size: int = 50,
        pad_context: bool = False,
    ) -> Iterable[tuple[str, str, str, str]]:
        """Generate KWIC results for multiple keywords in the document."""
        all_kwic_results = []
        for original_kw in keywords:
            for left, found_keyword, right in kwic.keyword_in_context(
                doc=doc,
                keyword=original_kw,
                ignore_case=ignore_case,
                window_width=window_size,
                pad_context=pad_context,
            ):
                all_kwic_results.append((left, found_keyword, right, str(original_kw)))

        return all_kwic_results

    def find_in_sentences(
        doc: Doc,  # Must be a Doc object, used for seperating sentences
        keyword: str | Pattern,
        ignore_case: bool = True,
    ) -> Iterable[tuple[str, str, str]]:
        """Generate KWIC results for a keyword in each sentence of the document."""
        if not isinstance(doc, Doc):
            raise LexosException(
                "Input 'doc' must be a spaCy Doc object for sentence-level search."
            )
        all_sentence_kwic_results = []
        for sent_idx, sentence_span in enumerate(doc.sents):
            for left, found_keyword, right in kwic.keyword_in_context(
                doc=sentence_span.text,
                keyword=keyword,
                ignore_case=ignore_case,
                window_width=len(sentence_span.text) * 2,  # capture entire sentence
                pad_context=False,
            ):
                all_sentence_kwic_results.append((left, found_keyword, right))

        return all_sentence_kwic_results
