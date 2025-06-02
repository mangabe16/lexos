"""record.py.

Last updated: June 2, 2025
Last tested: It works in a noteboook, but no unit tests written yet.

Wrapping texts and spaCy Docs in a Pydantic model provides a lot of extra functionality, particularly through the model_dump() and model_dump_json() methods. See the Pydantic documentation for more information.

Other than that, the Record class provides methods for serializing and deserializing the record to and from bytes, saving and loading the record to and from disk, and calculating various statistics about the record, such as the number of terms, tokens, vocabulary density, and most/least common terms.

The Record class handles the difficult task of keeping track of whether the content is a spaCy Doc or a string, as well as the tricky job of preserving custom Token attributes when spaCy Docs are serialised and deserialised.

This code is designed to work by default with UUID4 for the ID field, which is a universally unique identifier. UUID7 is a better choice but does not yet have full support in the Python standard library and Pydantic. Once that takes place, it can be easily changed in the Record model. Alternaively, the ID can be set to an incrementing integer with `id_type="integer"`.

# TODO:
- Test.
"""

import uuid
from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import msgpack
import spacy
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    computed_field,
    field_serializer,
    validate_call,
)
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Token

from lexos.corpus.utils import LexosModelCache
from lexos.exceptions import LexosException


class Record(BaseModel):
    """The main Record model."""

    id: int | UUID4 = uuid.uuid4()
    name: Optional[str] = None
    is_active: Optional[bool] = True
    content: Optional[Doc | str] = None
    model: Optional[str] = None
    extensions: Optional[list[str]] = []
    data_source: Optional[str] = None
    meta: Optional[dict[str, Any]] = {}

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        json_schema_extra=DocJSONSchema.schema(),
    )

    @field_serializer("content")
    def serialize_content(self, content: Doc | str):
        """Serialize the content to bytes if it is a Doc object.

        Args:
            content (Doc | str): The content to serialize.

        Returns:
            bytes | str: The serialized content as bytes if it is a Doc, otherwise the original string.
        """
        if isinstance(content, Doc):
            content.user_data["extensions"] = {}
            for ext in self.extensions:
                content.user_data["extensions"][ext] = [token._.get(ext) for token in content]
            return content.to_bytes()
        return content

    def __repr__(self):
        """Return a string representation of the record."""
        fields = self.model_dump(exclude=["terms", "text", "tokens"])
        fields["is_parsed"] = str(self.is_parsed)
        if self.content and self.is_parsed:
            fields["content"] = f"{self.content.text[:25]}..."
        elif self.content and not self.is_parsed:
            fields["content"] = f"{self.content[:25]}..."
        else:
            fields["content"] = "None"
        field_list = [f"{k}={v}" if v else f"{k}=None" for k, v in fields.items()]
        return f"Record({', '.join(field_list)})"

    @computed_field
    @cached_property
    def is_parsed(self) -> bool:
        """Return whether the record is parsed."""
        if isinstance(self.content, Doc):
            return True
        return False

    @computed_field
    @cached_property
    def preview(self) -> str:
        """Return a preview of the record text."""
        if self.content is None:
            return None

        if self.is_parsed:
            return f"{self.content.text[0:50]}..."
        return f"{self.content[0:500]}..."

    @computed_field
    @cached_property
    def terms(self) -> Counter:
        """Return the terms in the record."""
        if self.is_parsed:
            return Counter([t.text for t in self.content])
        else:
            raise LexosException("Record is not parsed.")

    @property
    def text(self) -> str:
        """Return the text of the record."""
        if self.is_parsed:
            return self.content.text
        return self.content

    @cached_property
    def tokens(self) -> list[str]:
        """Return the tokens in the record."""
        if self.is_parsed:
            return [t.text for t in self.content]
        else:
            raise LexosException("Record is not parsed.")

    def _doc_from_bytes(
        self,
        content: bytes,
        model: Optional[str] = None,
        model_cache: Optional[LexosModelCache] = None,
    ) -> Doc:
        """Convert bytes to a Doc object.

        Args:
            content (bytes): The bytes to convert.
            model (Optional[str]): The spaCy model to use for loading the Doc.
            model_cache (Optional[LexosModelCache]): An optional cache for spaCy models.

        Returns:
            Doc: The content as a Doc object.
        """
        # Create a Doc from the bytes
        vocab = self._get_vocab(model, model_cache)
        doc = Doc(vocab).from_bytes(content)

        # Restore extension values
        for ext, values in doc.user_data["extensions"].items():
            Token.set_extension(ext, default=None, force=True)
            for i in range(len(doc)):
                doc[i]._.set(ext, values[i])

        # Clean up user_data
        doc.user_data["extensions"] = list(doc.user_data["extensions"].keys())

        return doc

    # WARNING: This method is deprecated in favour of field serializer.
    def _doc_to_bytes(self) -> bytes:
        """Convert the content to bytes if it is a Doc object.

        Returns:
            bytes: The content as bytes.
        """
        if not isinstance(self.content, Doc):
            raise LexosException("Content is not a Doc object.")

        doc = self.content

        doc.user_data["extensions"] = {}
        for ext in self.extensions:
            doc.user_data["extensions"][ext] = [token._.get(ext) for token in doc]

        return doc.to_bytes()

    def _get_vocab(
        self, model: Optional[str] = None, model_cache: Optional[LexosModelCache] = None
    ):
        """Get the vocabulary from the model or model cache.

        Args:
            model (Optional[str]): The spaCy model to use for loading the Doc.
            model_cache (Optional[LexosModelCache]): An optional cache for spaCy models.

        Returns:
            Vocab: The vocabulary of the model.
        """
        if model_cache and not model:
            raise LexosException("Model cache provided but no model specified.")

        if model_cache:
            return model_cache.get_model(model).vocab
        elif model:
            return spacy.load(model).vocab
        elif self.model:
            return spacy.load(self.model).vocab
        else:
            raise LexosException(
                "No model specified for loading the Doc. Please provide a model name or a model cache."
            )

    @validate_call(config=model_config)
    def from_bytes(
        self,
        bytestring: bytes,
        model: Optional[str] = None,
        model_cache: Optional[LexosModelCache] = None,
    ) -> None:
        """Deserialise the record from bytes.

        Args:
            bytestring (bytes): The bytes to load the record from.
        """
        data = msgpack.unpackb(bytestring)

        # Update the record with the loaded data
        for k, v in data.items():
            if k in self.model_fields:
                if k != "content":
                    setattr(self, k, v)

        # If content is bytes, convert it back to a Doc object
        if data["is_parsed"] and isinstance(data["content"], bytes):
            if not model:
                model = data.get("model")
            self.content = self._doc_from_bytes(data["content"], model, model_cache)

    @validate_call(config=model_config)
    def from_disk(
        self,
        path: Path | str,
        model: Optional[str] = None,
        model_cache: Optional[LexosModelCache] = None,
    ) -> None:
        """Load the record from disk.

        Args:
            path (Path | str): The path to load the record from.
            model (Optional[str]): The spaCy model to use for loading the Doc.
            model_cache (Optional[LexosModelCache]): An optional cache for spaCy models.
        """
        if not path:
            raise LexosException("No path specified for loading the record.")

        # Load the data from disk
        with open(path, "rb") as f:
            # data = msgpack.unpack(f)
            data = f.read()

        # Get the record content from the bytestring
        self.from_bytes(data, model=model, model_cache=model_cache)

    def least_common_terms(self, n: Optional[int] = None) -> int:
        """Return the least common terms.

        Args:
            n (Optional[int]): The number of least common terms to return. If None, return all terms.

        Returns:
            int: The least common terms in the record.
        """
        if self.is_parsed:
            return (
                sorted(self.terms.items(), key=lambda x: x[1])[:n]
                if n
                else sorted(self.terms.items(), key=lambda x: x[1])
            )
        else:
            raise LexosException("Record is not parsed.")

    def most_common_terms(self, n: Optional[int] = None) -> int:
        """Return the most common terms.

        Args:
            n (Optional[int]): The number of most common terms to return. If None, return all terms.

        Returns:
            int: The most common terms in the record.
        """
        if self.is_parsed:
            return self.terms.most_common(n)
        else:
            raise LexosException("Record is not parsed.")

    def num_terms(self) -> int:
        """Return the number of terms."""
        if self.is_parsed:
            return len(self.terms)
        else:
            raise LexosException("Record is not parsed.")

    def num_tokens(self) -> int:
        """Return the number of tokens."""
        if self.is_parsed:
            return len(self.content)
        else:
            raise LexosException("Record is not parsed.")

    @validate_call(config=model_config)
    def to_bytes(self, extensions: Optional[list[str]] = []) -> bytes:
        """Serialize the record to a dictionary.

        Args:
            extensions (list[str]): A list of extension names to include in the serialization.

        Returns:
            bytes: The serialized record.
        """
        # Handle extensions
        if extensions:
            self.extensions = list(set(self.extensions + extensions))

        # Convert record to a dictionary
        data = self.model_dump(exclude=["terms", "text", "tokens"])

        # Make UUID serialisable
        data["id"] = str(data["id"])

        # WARNING: This code is deprecated in favour of field serializer.
        # Convert the content to bytes if it is a Doc object
        if self.is_parsed:
            data["content"] = self._doc_to_bytes()

        return msgpack.dumps(data)

    @validate_call(config=model_config)
    def to_disk(self, path: Path | str, extensions: Optional[list[str]] = None) -> None:
        """Save the record to disk.

        Args:
            path (Path | str): The path to save the record to.
            extensions (list[str]): A list of extension names to include in the serialization.
        """
        if not path:
            raise LexosException("No path specified for saving the record.")

        if not extensions:
            extensions = self.extensions

        # Serialize and save the record
        data = self.to_bytes(extensions)
        with open(path, "wb") as f:
            f.write(data)

    def vocab_density(self) -> float:
        """Return the vocabulary density.

        Returns:
            float: The vocabulary density of the record.
        """
        if self.is_parsed:
            return self.num_terms() / self.num_tokens()
        else:
            raise LexosException("Record is not parsed.")
