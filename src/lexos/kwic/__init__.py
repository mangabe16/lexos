"""__init__.py.

Last Updated: 6/27/25
Last Tested: 6/27/25

Current Usage:
- Find keywords and their surrounding context in spaCy docs
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
from lexos.util import ensure_list


class Kwic:
    """A class for generating keyword-in-context (KWIC) results using textacy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    nlp = spacy.load("xx_sent_ud_sm")

    @validate_call(config=model_config)
    def find(
        doc: Doc | list[Doc],
        keyword: str | Pattern,
        ignore_case: bool = True,
        window_size: int = 50,
        pad_context: bool = False,
        dataframe_format: bool = False,
    ) -> Iterable[tuple[str, str, str]] | pd.DataFrame:
        """Generate KWIC results for a given term in the document.

        Args:
            doc (Doc | Iterable[Doc]): The spaCy Doc or list of Docs to search within.
            keyword (str | Pattern): The keyword to search for, can be a string or a regex pattern.
            ignore_case (bool): Whether to ignore case when searching for the keyword.
            window_size (int): The number of characters to include as context on either side of the keyword.
            pad_context (bool): Whether to pad the context with empty space if the keyword is at the start or end of the document.
            dataframe_format (bool): Whether to return the results in the form of a pandas DataFrame.

        Returns:
            Iterable[tuple[str, str, str]]: An iterable of tuples containing the left context, the keyword, and the right context.
            pd.DataFrame: A DataFrame containing the left context, the keyword, and the right context if dataframe_format is True.

        """
        ret = []
        for eachDoc in ensure_list(doc):
            ret.append(
                list(
                    kwic.keyword_in_context(
                        doc=eachDoc,
                        keyword=keyword,
                        ignore_case=ignore_case,
                        window_width=window_size,
                        pad_context=pad_context,
                    )
                )
            )

        if dataframe_format:
            # from RomanPerekhrest on StackOverflow
            # https://stackoverflow.com/questions/57509968/list-of-lists-of-tuples-to-pandas-dataframe
            return pd.DataFrame(
                [t for lst in ret for t in lst], columns=["Left", "Keyword", "Right"]
            )
        return ret

    @validate_call(config=model_config)
    def find_multiple_keywords(
        doc: Doc | Iterable[Doc],
        keywords: Iterable[str | Pattern],
        ignore_case: bool = True,
        window_size: int = 50,
        pad_context: bool = False,
        dataframe_format: bool = False,
    ) -> Iterable[tuple[str, str, str, str]] | pd.DataFrame:
        """Generate KWIC results for multiple keywords in the document.

        Args:
            doc (Doc | Iterable[Doc]): The spaCy Doc or list of Docs to search within.
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
        for doc in ensure_list(doc):
            for original_kw in keywords:
                for left, found_keyword, right in kwic.keyword_in_context(
                    doc=doc,
                    keyword=original_kw,
                    ignore_case=ignore_case,
                    window_width=window_size,
                    pad_context=pad_context,
                ):
                    all_kwic_results.append(
                        (left, found_keyword, right, str(original_kw))
                    )

        if dataframe_format:
            return pd.DataFrame(
                all_kwic_results,
                columns=["Left", "Keyword", "Right", "Original Keyword"],
            )
        return all_kwic_results

    @validate_call(config=model_config)
    def find_in_sentences(
        doc: Doc | Iterable[Doc],
        keyword: str | Pattern,
        ignore_case: bool = True,
        dataframe_format: bool = False,
    ) -> Iterable[tuple[str, str, str]] | pd.DataFrame:
        """Generate KWIC results for a keyword in each sentence of the document.

        Args:
            doc (Doc | Iterable[Doc]): The spaCy Doc or list of Docs to search within.
            keyword (str | Pattern): The keyword to search for, can be a string or a regex pattern.
            ignore_case (bool): Whether to ignore case when searching for the keyword.
            dataframe (bool): Whether to return the results in the form of a pandas DataFrame.

        Returns:
            Iterable[tuple[str, str, str]]: An iterable of tuples containing the left context, the keyword found, and the right context for each sentence.
            pd.DataFrame: A DataFrame containing the left context, the keyword found, and the right context if dataframe_format is True.
        """
        all_sentence_kwic_results = []
        for doc in ensure_list(doc):
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
        doc: Doc | Iterable[Doc],
        keyword: str | Pattern,
        token_window: int = 5,
        ignore_case: bool = True,
        dataframe_format: bool = False,
        nlp: Language = nlp,
    ) -> Iterable[tuple[str, str, str]] | pd.DataFrame:
        """Generate KWIC results for a keyword within documents, using a window of tokens as context.

        Args:
            doc (Doc | Iterable[Doc]): The spaCy Doc or list of Docs to search within.
            keyword (str | Pattern): The keyword to search for, can be a string or a regex pattern.
            token_window (int): The number of tokens to include as context on each side of the keyword.
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
        if isinstance(keyword, str):
            if ignore_case:
                matcher.add("search", [[{"LOWER": keyword.lower()}]])
            else:
                matcher.add("search", [[{"TEXT": keyword}]])
        else:
            pattern = [{"TEXT": {"REGEX": keyword.pattern}}]
            matcher.add("search", [pattern])

        # Get the matches in the document(s)
        all_matches = []
        for doc in ensure_list(doc):
            matches = matcher(doc)
            for match in matches:
                all_matches.append((doc, match))

        # If there are no matches, return an empty list or DataFrame
        if not all_matches:
            raise LexosException(f"No matches found for keyword: {keyword}")

        hits = []
        # Iterate over the matches
        for doc_of_match, match in all_matches:
            match_id, start, end = match  # Get the start and end indices of the match
            # Get the matched span.
            span = doc_of_match[start:end]  # The matched span

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
