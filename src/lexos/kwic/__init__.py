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
    def find_to_df(
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
            pad_context=pad_context),
            columns=["Left", "Keyword", "Right"]
        )
