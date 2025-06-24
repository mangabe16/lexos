"""__init__.py.

Last Updated: 6/24/25
Last Tested: 6/24/25

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
from spacy.matcher import Matcher
from spacy.language import Language
import spacy

from lexos.exceptions import LexosException


class Kwic:
    """A class for generating keyword-in-context (KWIC) results using textacy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    nlp = spacy.load("xx_sent_ud_sm")

    @validate_call(config=model_config)
    def find(
        doc: Doc | str,
        keyword: str | Pattern,
        ignore_case: bool = True,
        window_size: int = 50,
        pad_context: bool = False,
        dataframe_format: bool = False,
    ) -> Iterable[tuple[str, str, str]] | pd.DataFrame:
        """Generate KWIC results for a given term in the document.

        Args:
            doc (Doc | str): The document to search within, either as a spaCy Doc or a string.
            keyword (str | Pattern): The keyword to search for, can be a string or a regex pattern.
            ignore_case (bool): Whether to ignore case when searching for the keyword.
            window_size (int): The number of characters to include as context on either side of the keyword.
            pad_context (bool): Whether to pad the context with empty space if the keyword is at the start or end of the document.
            dataframe_format (bool): Whether to return the results in the form of a pandas DataFrame.

        Returns:
            Iterable[tuple[str, str, str]]: An iterable of tuples containing the left context, the keyword, and the right context.
            pd.DataFrame: A DataFrame containing the left context, the keyword, and the right context if dataframe_format is True.

        """
        ret = kwic.keyword_in_context(
            doc=doc,
            keyword=keyword,
            ignore_case=ignore_case,
            window_width=window_size,
            pad_context=pad_context,
        )

        if dataframe_format:
            return pd.DataFrame(ret, columns=["Left", "Keyword", "Right"])
        return ret

    @validate_call(config=model_config)
    def find_multiple_keywords(
        doc: Doc | str,
        keywords: Iterable[str | Pattern],
        ignore_case: bool = True,
        window_size: int = 50,
        pad_context: bool = False,
        dataframe_format: bool = False,
    ) -> Iterable[tuple[str, str, str, str]] | pd.DataFrame:
        """Generate KWIC results for multiple keywords in the document.

        Args:
            doc (Doc | str): The document to search within, either as a spaCy Doc or a string.
            keywords (Iterable[str | Pattern]): An iterable of keywords to search for, can be strings or regex patterns.
            ignore_case (bool): Whether to ignore case when searching for the keywords.
            window_size (int): The number of characters to include as context on either side of the keyword.
            pad_context (bool): Whether to pad the context with empty space if the keyword is at the start or end of the document.
            dataframe_format (bool): Whether to return the results in the form of a pandas DataFrame.

        Returns:
            Iterable[tuple[str, str, str, str]]: An iterable of tuples containing the left context, the keyword found, the right context, and the original keyword for each search.
            pd.DataFrame: A DataFrame containing the left context, the keyword found, the right context, and the original keyword if dataframe_format is True.
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

        if dataframe_format:
            return pd.DataFrame(
                all_kwic_results,
                columns=["Left", "Keyword", "Right", "Original Keyword"],
            )
        return all_kwic_results

    @validate_call(config=model_config)
    def find_in_sentences(
        doc: Doc,  # Must be a Doc object, used for seperating sentences
        keyword: str | Pattern,
        ignore_case: bool = True,
        dataframe_format: bool = False,
    ) -> Iterable[tuple[str, str, str]] | pd.DataFrame:
        """Generate KWIC results for a keyword in each sentence of the document.

        Args:
            doc (Doc): The spaCy Doc object containing the text to search within.
            keyword (str | Pattern): The keyword to search for, can be a string or a regex pattern.
            ignore_case (bool): Whether to ignore case when searching for the keyword.
            dataframe (bool): Whether to return the results in the form of a pandas DataFrame.

        Returns:
            Iterable[tuple[str, str, str]]: An iterable of tuples containing the left context, the keyword found, and the right context for each sentence.
            pd.DataFrame: A DataFrame containing the left context, the keyword found, and the right context if dataframe_format is True.
        """
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

        if dataframe_format:
            return pd.DataFrame(
                all_sentence_kwic_results,
                columns=["Left", "Keyword", "Right"],
            )
        return all_sentence_kwic_results

    @validate_call(config=model_config)
    def find_tokens(
        doc: Doc,
        keyword: str | Pattern,
        token_window: int = 5,
        ignore_case: bool = True,
        dataframe_format: bool = False,
        nlp: Language = nlp
    ) -> Iterable[tuple[str, str, str]] | pd.DataFrame:
        """Generate KWIC results for a keyword in each sentence of the document, using a window of tokens as context.

        Args:
            doc (Doc): The spaCy Doc object containing the text to search within.
            keyword (str | Pattern): The keyword to search for, can be a string or a regex pattern.
            token_window (int): The number of tokens to include as context on each side.
            ignore_case (bool): Whether to ignore case when searching for the keyword.
            dataframe_format (bool): Whether to return the results in the form of a pandas DataFrame.
            nlp (Language): The spaCy Language model to use for matching.

        Returns:
            Iterable [tuple[str, str, str]]: An iterable of tuples containing the left context, the keyword found, and the right context for each sentence.
            pd.DataFrame: A DataFrame containing the left context, the keyword found, and the right context if dataframe_format is True.
        """
        # Instantiate the Matcher with the Doc's vocabulary
        matcher = Matcher(nlp.vocab)
        # Add the keyword pattern to the matcher
        if (ignore_case):
            matcher.add("search", [[{"LOWER": keyword.lower()}]])
        else:
            matcher.add("search", [[{"TEXT": keyword}]])
        # Get the matches in the document
        matches = matcher(doc)

        # If there are no matches, return an empty list or DataFrame
        if not matches:
            raise LexosException(f"No matches found for keyword: {keyword}")

        hits = []
        # Iterate over the matches
        for match_id, start, end in matches:
            # Get the matched span.
            span = doc[start:end]  # The matched span
            left = start - token_window
            right = end + token_window
            left = max(left, 0)  # Ensure we don't go out of bounds
            right = min(right, len(doc))  # Ensure we don't go out of bounds
            left_context = doc[left:start].text.strip()
            right_context = doc[end:right].text.strip()
            hits.append((left_context, span.text, right_context))

        if dataframe_format:
            return pd.DataFrame(hits, columns=["Left", "Keyword", "Right"])

        return hits
