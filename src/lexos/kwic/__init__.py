"""__init__.py."""

from textacy.extract import kwic
from spacy.tokens import Doc
from typing import Iterable, Pattern
from pydantic import validate_call

from lexos.exceptions import LexosException


class Kwic:
    """A class for generating keyword-in-context (KWIC) results using textacy."""

    #@validate_call
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
