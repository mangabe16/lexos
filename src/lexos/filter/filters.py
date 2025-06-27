"""filters.py.

Last Update: June 24, 2025
Last Tested: June 24, 2025

The filter model provides a base class for applying filters to a document and returning
a new document, as well as extracting tokens or ids form filtered docs. Filters are
applied by passing a spaCy matcher to identify matches to the filter criteria. Since the
doc's language model may not supply the required token attributes, you can create a custom
filter to add and set the attributes as custom extensions. Examples of custom filter classes
are `IsRomanFilter` and `IsWordFilter`.

The module also provides a useful `StopwordFilter` class to add or remove stop words from a
spaCy doc without retokenising. Note, however, that it works by changing the model's defaults,
so they will apply to any doc created with that model unless the model is reloaded.
"""

import re
from typing import Any, ClassVar, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.matcher import Matcher
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Token

from lexos.exceptions import LexosException
from lexos.util import ensure_list


class BaseFilter(BaseModel):
    """BaseFilter class."""

    id: ClassVar[str] = "base_filter"
    doc: Optional[Doc] = None
    matcher: Optional[Matcher] = None
    matches: Optional[list[tuple[int, int, int]]] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    @validate_call(config=model_config)
    def __call__(self, doc: Optional[Doc], matcher: Optional[Matcher] = None) -> Doc:
        """Call the filter function."""
        # Validate the inputs
        if not doc and not self.doc:
            raise LexosException("No doc has been assigned to the filter.")
        if not matcher and not self.matcher:
            raise LexosException("No matcher has been assigned to the filter.")
        if doc:
            self.doc = doc
        if matcher:
            self.matcher = matcher
        # Get the matches
        self.matches = self.matcher(doc)

    @property
    def matched_token_ids(self) -> set[int]:
        """A list of matched token ids after the filter was applied."""
        if not self.matches:
            return None
        token_ids = set()
        for _, start, end in self.matches:
            for i in range(start, end):
                token_ids.add(i)
        return token_ids

    @property
    def matched_tokens(self) -> list[int]:
        """A list of matched tokens after the filter was applied."""
        return [self.doc[i] for i in self.matched_token_ids]

    @property
    def filtered_tokens(self) -> list[int]:
        """A list of filtered tokens after the filter was applied."""
        return [self.doc[i] for i in self.filtered_token_ids]

    @property
    def filtered_token_ids(self) -> set[int]:
        """A list of filtered token ids after the filter was applied."""
        if not self.matches:
            return None
        return set(range(len(self.doc))) - self.matched_token_ids

    def _set_extensions(self, attr: str, default: Any):
        """Set the extensions."""
        if not Token.has_extension(attr):
            Token.set_extension(attr, default=default, force=True)

    def get_matched_doc(self) -> Doc:
        """Get a new doc from the matched tokens."""
        words = [t.text for t in self.matched_tokens]
        spaces = [t.whitespace_ for t in self.matched_tokens]
        return Doc(self.doc.vocab, words=words, spaces=spaces)

    def get_filtered_doc(self) -> Doc:
        """Get a new doc from the matched tokens."""
        words = [
            t.text for t in self.matched_tokens if t.i not in self.filtered_token_ids
        ]
        spaces = [
            t.whitespace_
            for t in self.matched_tokens
            if t.i not in self.filtered_token_ids
        ]
        return Doc(self.doc.vocab, words=words, spaces=spaces)


class IsRomanFilter(BaseFilter):
    """A filter for Roman numerals."""

    id: ClassVar[str] = "is_roman"
    doc: Optional[Doc] = None
    attr: Optional[str] = None
    default: Optional[Any] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    def __init__(self, **data):
        """Initialise the filter object and set custom attribute extensions."""
        super().__init__(**data)
        if self.attr:
            self._set_extensions(self.attr, self.default)

    @validate_call(config=model_config)
    def __call__(
        self,
        doc: Optional[Doc],
        attr: Optional[str] = None,
        default: Optional[Any] = None,
    ) -> Doc:
        """Apply the filter.

        Args:
            doc (Optional[Doc]): A spaCy doc.
            attr (Optional[str]): The name of the attribute to add to the tokens.
            default (Optional[Any]): The default value of the attribute.

        Returns:
            Doc: The filtered doc.
        """
        # Validation
        if doc:
            self.doc = doc
        if attr:
            self.attr = attr
        if default is not None:
            self.default = default
        
        # Use instance attributes if we have them
        working_doc = self.doc if self.doc is not None else doc
        working_attr = self.attr if hasattr(self, 'attr') and self.attr is not None else attr
        working_default = self.default if hasattr(self, 'default') and self.default is not None else default
        
        # Set custom extensions
        if working_attr:
            self._set_extensions(working_attr, working_default)
        
        # Apply the filter only if we have a valid doc
        if working_doc is not None:
            for i, token in enumerate(working_doc):
                setattr(working_doc[i]._, working_attr, self.is_roman(token))

        return working_doc if working_doc is not None else doc

    @validate_call(config=model_config)
    def is_roman(self, token: Token) -> bool:
        """Detect Roman numerals (capitals only).

        Args:
            token (Token): A spaCy token.

        Returns:
            bool: True if the token is a Roman numeral.
        """
        if token.text == "":
            return False
        pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
        return bool(re.search(pattern, token.text))


class IsStopwordFilter(BaseFilter):
    """A filter to detect stop words in a spaCy doc."""

    id: ClassVar[str] = "is_stopword"
    doc: Optional[Doc] = None
    stopwords: Optional[list | str] = None
    remove: Optional[bool] = False

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    def __init__(self, **data):
        """Initialise the filter object with configuration.

        Args:
            doc (Optional[Doc]): A spaCy doc.
            stopwords (Optional[list | str]): A list or string containing the stop word(s) to add or remove.
            remove (Optional[bool]): If True, the stop word(s) will be removed from the model.
        """
        super().__init__(**data)
        self.stopwords = ensure_list(self.stopwords)

    @validate_call(config=model_config)
    def __call__(
        self,
        doc: Optional[Doc],
        stopwords: Optional[list | str] = None,
        remove: Optional[bool] = False,
    ) -> Doc:
        """Apply the filter.

        Args:
            doc (Optional[Doc]): A spaCy doc.
            stopwords (Optional[list | str]): A list or string containing the stop word(s) to add or remove.
            default (Optional[Any]): If True, the stop word(s) will be removed from the model.

        Returns:
            Doc: The filtered doc.

        Note:
            This filter modifies the model defaults. If you need the model's original default stop words.
            you will need to re-load the model.
        """
        # Validation
        if doc:
            self.doc = doc
        if stopwords is not None:
            self.stopwords = ensure_list(stopwords)
        if remove is not None:
            self.remove = remove
        
        # Use instance attributes if parameters are None
        working_doc = self.doc if self.doc is not None else doc
        # Handle stopwords carefully - convert to list if it's a pydantic ValidatorIterator
        if stopwords is None and hasattr(self, 'stopwords') and self.stopwords is not None:
            try:
                working_stopwords = list(self.stopwords)
            except (TypeError, AttributeError):
                working_stopwords = self.stopwords
        else:
            working_stopwords = stopwords
        working_remove = self.remove if hasattr(self, 'remove') and self.remove is not None else remove
        
        # Apply the filter only if we have valid inputs
        if working_doc is not None and working_stopwords is not None:
            # Ensure stopwords is iterable and properly formatted
            if not isinstance(working_stopwords, list):
                working_stopwords = ensure_list(working_stopwords)
            
            if working_remove:
                for item in working_stopwords:
                    if item is not None:  # Skip None values
                        working_doc.vocab[item].is_stop = False
            else:
                for item in working_stopwords:
                    if item is not None:  # Skip None values
                        working_doc.vocab[item].is_stop = True

        return working_doc if working_doc is not None else doc


class IsWordFilter(BaseFilter):
    """A filter to detect words in a spaCy doc."""

    id: ClassVar[str] = "is_word"
    doc: Optional[Doc] = None
    attr: Optional[str] = "is_word"
    default: Optional[bool] = False
    exclude: Optional[Optional[list[str] | str]] = [" ", "\n"]
    exclude_digits: Optional[Optional[bool]] = False
    exclude_roman_numerals: Optional[Optional[bool]] = False
    exclude_pattern: Optional[list[str] | str] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    def __init__(self, **data):
        """Initialise the filter object with configuration.

        Args:
            doc (Optional[Doc]): A spaCy doc.
            attr (Optional[str]): The name of the attribute to add to the tokens.
            default (Optional[Any]): The default value of the attribute.
            exclude (Optional[list[str] | str]): A string/regex or list of strings/regex patterns to exclude.
            exclude_digits: (Optional[bool]): If True, digits will not be treated as words.
            exclude_roman_numerals (Optional[bool]): Same as above for Roman numerals, but only works on capital letters.
            exclude_pattern (Optional[list[str] | str]): Additional patterns to add to the default exclude list.
        """
        super().__init__(**data)
        if self.attr:
            self._set_extensions(self.attr, self.default)

    @validate_call(config=model_config)
    def __call__(
        self,
        doc: Optional[Doc],
        attr: Optional[str] = "is_word",
        default: Optional[bool] = False,
        exclude: Optional[list[str] | str] = [" ", "\n"],
        exclude_digits: Optional[bool] = False,
        exclude_roman_numerals: Optional[bool] = False,
        exclude_pattern: Optional[list[str] | str] = None,
    ) -> Doc:
        """Apply the filter.

        Args:
            doc (Optional[Doc]): A spaCy doc.
            attr (Optional[str]): The name of the attribute to add to the tokens.
            default (Optional[Any]): The default value of the attribute.
            exclude (Optional[list[str] | str]): A string/regex or list of strings/regex patterns to exclude.
            exclude_digits: (Optional[bool]): If True, digits will not be treated as words.
            exclude_roman_numerals (Optional[bool]): Same as above for Roman numerals, but only works on capital letters.
            exclude_pattern (Optional[list[str] | str]): Additional patterns to add to the default exclude list.

        Returns:
            Doc: The filtered doc.
        """
        # Assign keyword variables to the instance attributes
        if doc:
            self.doc = doc
        if exclude:
            self.exclude = ensure_list(exclude)
        if exclude_digits is not None:
            self.exclude_digits = exclude_digits
        if exclude_roman_numerals is not None:
            self.exclude_roman_numerals = exclude_roman_numerals
        if exclude_pattern:
            self.exclude_pattern = ensure_list(exclude_pattern)
        if attr:
            self.attr = attr
        if default is not None:
            self.default = default
        
        # Use instance attributes if we have them
        working_doc = self.doc if self.doc is not None else doc
        working_attr = self.attr if hasattr(self, 'attr') and self.attr is not None else attr
        working_default = self.default if hasattr(self, 'default') and self.default is not None else default
        
        # Set ._is_word extension
        if working_attr:
            self._set_extensions(working_attr, working_default)
        
        # Apply the filter only if we have a valid doc
        if working_doc is not None:
            for i, token in enumerate(working_doc):
                setattr(working_doc[i]._, working_attr, self.is_word(token))

        return working_doc if working_doc is not None else doc

    def _is_roman_numeral(self, string: str) -> bool:
        """Check if a string is a Roman numeral.

        Args:
            string (str): A string.

        Returns:
            bool: True if the string is a Roman numeral.
        """
        if string == "":
            return False
        pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
        return bool(re.search(pattern, string))

    @validate_call(config=model_config)
    def is_word(self, token: Token) -> bool:
        """Detect words.

        Args:
            token (Token): A spaCy token.

        Returns:
            bool: True if the token is a word.
        """
        predicates = []
        if self.exclude_digits:
            predicates.append(lambda token: token.text.isalpha())
        else:
            predicates.append(
                lambda token: token.text.isalpha() or token.text.isdigit()
            )
        if self.exclude_roman_numerals:
            predicates.append(lambda token: not self._is_roman_numeral(token.text))
        if self.exclude_pattern:
            self.exclude += self.exclude_pattern
        if len(self.exclude) > 0:
            exclude_pat = "|".join(self.exclude)
            predicates.append(lambda token: re.search(exclude_pat, token.text) is None)
        return all([f(token) for f in predicates])
