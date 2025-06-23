"""__init__.py.

Last Updated: 6/23/25
Last Tested: 6/18/25

Current Usage:
- Find keywords within their context in a spaCy doc or string
    - Returns as either an iterable of tuples or a pandas DataFrame
- Find multiple keywords and their context in a spaCy doc or string
- Find keywords in sentences of a spaCy doc
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
        """Generate KWIC results for a given term in the document.

        Args:
            doc (Doc | str): The document to search within, either as a spaCy Doc or a string.
            keyword (str | Pattern): The keyword to search for, can be a string or a regex pattern.
            ignore_case (bool): Whether to ignore case when searching for the keyword.
            window_size (int): The number of characters to include as context on either side of the keyword.
            pad_context (bool): Whether to pad the context with empty space if the keyword is at the start or end of the document.

        Returns:
            Iterable[tuple[str, str, str]]: An iterable of tuples containing the left context, the keyword, and the right context.

        """
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
        """Generate KWIC results for a given term in the document in the form of a pandas DataFrame.

        Args:
            doc (Doc | str): The document to search within, either as a spaCy Doc or a string.
            keyword (str | Pattern): The keyword to search for, can be a string or a regex pattern.
            ignore_case (bool): Whether to ignore case when searching for the keyword.
            window_size (int): The number of characters to include as context on either side of the keyword.
            pad_context (bool): Whether to pad the context with empty space if the keyword is at the start or end of the document.

        Returns:
            pd.DataFrame: A DataFrame containing the left context, the keyword, and the right context.
        """
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
        """Generate KWIC results for multiple keywords in the document.

        Args:
            doc (Doc | str): The document to search within, either as a spaCy Doc or a string.
            keywords (Iterable[str | Pattern]): An iterable of keywords to search for, can be strings or regex patterns.
            ignore_case (bool): Whether to ignore case when searching for the keywords.
            window_size (int): The number of characters to include as context on either side of the keyword.
            pad_context (bool): Whether to pad the context with empty space if the keyword is at the start or end of the document.

        Returns:
            Iterable[tuple[str, str, str, str]]: An iterable of tuples containing the left context, the keyword found, the right context, and the original keyword for each search.
        """
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

    @validate_call(config=model_config)
    def find_in_sentences(
        doc: Doc,  # Must be a Doc object, used for seperating sentences
        keyword: str | Pattern,
        ignore_case: bool = True,
    ) -> Iterable[tuple[str, str, str]]:
        """Generate KWIC results for a keyword in each sentence of the document.

        Args:
            doc (Doc): The spaCy Doc object containing the text to search within.
            keyword (str | Pattern): The keyword to search for, can be a string or a regex pattern.
            ignore_case (bool): Whether to ignore case when searching for the keyword.

        Returns:
            Iterable[tuple[str, str, str]]: An iterable of tuples containing the left context, the keyword found, and the right context for each sentence.
        """
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

    @validate_call(config=model_config)
    def find_tokens(
        doc: Doc,
        keyword: str | Pattern,
        token_window: int = 5,
        ignore_case: bool = True,
    ) -> Iterable[tuple[str, str, str]]:
        """Generate KWIC results for a keyword in each sentence of the document, using a window of tokens as context.

        Args:
            doc (Doc): The spaCy Doc object containing the text to search within.
            keyword (str | Pattern): The keyword to search for, can be a string or a regex pattern.
            token_window (int): The number of tokens to include as context on each side.
            ignore_case (bool): Whether to ignore case when searching for the keyword.

        Returns:
            Iterable [tuple[str, str, str]]: An iterable of tuples containing the left context,
            the keyword found, and the right context for each sentence.
        """
        if not isinstance(doc, Doc):
            raise LexosException(
                "Input 'doc' must be a spaCy Doc object for sentence-level search."
            )
        all_sentence_kwic_results = []
        for sentence_span in doc.sents:
            tokens = list(sentence_span)
            for i, token in enumerate(tokens):
                # Check if token matches keyword (string or pattern)
                if (
                    isinstance(keyword, str)
                    and (
                        (token.text.lower() == keyword.lower())
                        if ignore_case
                        else (token.text == keyword)
                    )
                ) or (hasattr(keyword, "search") and keyword.search(token.text)):
                    left = " ".join(
                        t.text for t in tokens[max(0, i - token_window) : i]
                    )
                    right = " ".join(
                        t.text for t in tokens[i + 1 : i + 1 + token_window]
                    )
                    all_sentence_kwic_results.append((left, token.text, right))
        return all_sentence_kwic_results
