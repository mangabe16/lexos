"""corpus.py.

Last updated: June 5, 2025
Last tested: It works in a noteboook, but no unit tests written yet.

This code is designed to work by default with UUID4 for the ID field, which is a universally unique identifier. UUID7 is a better choice but does not yet have full support in the Python standard library and Pydantic. Once that takes place, it can be easily changed in the Record model. Alternaively, the ID can be set to an incrementing integer with `id_type="integer"`.

To reproduce the web app's Statistics module, call `stats = Corpus.get_token_stats()` to get a `CorpusStats` object. Its method produce the web app's calculations and output. By default, the `get_token_stats()` method retrieves stats for the entire corpus, but you can pass parameters to filter active documents or settings accepted by the vectorizer. You can also pass an arbitrary list of tuples containing the record ID, name, and tokens to retrieve statistics for any list of pre-tokenised documents.


# TODO:
- Test.
"""

import shutil
import uuid
from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import srsly
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    validate_call,
)
from spacy.tokens import Doc

from lexos.corpus import Record
from lexos.corpus.corpus_stats import CorpusStats
from lexos.corpus.utils import LexosModelCache, RecordsDict
from lexos.exceptions import LexosException


class Corpus(BaseModel):
    """A collection of Record objects."""

    corpus_dir: str = Field(
        "corpus", description="The path to the directory where the corpus is stored."
    )
    corpus_metadata_file: str = Field(
        "corpus_metadata.json",
        description="The name of the corpus metadata file.",
    )
    name: str = Field(None, description="The name of the corpus.")
    records: RecordsDict = Field({}, description="Dictionary of records in the corpus.")
    names: list[str] = []
    meta: dict[str | Any] = Field(
        {},
        description="Metadata dictionary for arbitrary metadata relating to the corpus.",
    )
    model_cache: LexosModelCache = Field(
        LexosModelCache(),
        description="A cache for spaCy models used in the Corpus.",
    )
    num_active_docs: int = Field(
        0, description="Number of active documents in the corpus."
    )
    num_docs: int = Field(0, description="Total number of documents in the corpus.")
    num_terms: int = Field(0, description="Total number of unique terms in the corpus.")
    num_tokens: int = Field(0, description="Total number of tokens in the corpus.")
    terms: set = Field(set(), description="Set of unique terms in the corpus.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialise the Corpus with a data directory and a metadata file."""
        super().__init__(**data)
        corpus_dir = Path(self.corpus_dir)
        Path(corpus_dir / "data").mkdir(parents=True, exist_ok=True)
        srsly.write_json(corpus_dir / self.corpus_metadata_file, self.model_dump_json())
        print("Corpus created.")

    def __repr__(self):
        """Return a string representation of the Corpus."""
        fields = {field: getattr(self, field) for field in self.model_fields_set}
        field_list = [f"{k}={v}" for k, v in fields.items()]
        rep = f"Corpus({', '.join(sorted(field_list))})"
        return rep

    @property
    def active_terms(self) -> set:
        """Return the set of active terms in the Corpus."""
        active_terms = set()
        for record in self.records:
            if record.is_parsed and record.is_active:
                active_terms.update(record.terms.keys())
        return active_terms

    @property
    def meta_df(self) -> pd.DataFrame:
        """Return a DataFrame of the Corpus metadata."""
        if not self.meta:
            raise LexosException("No metadata available in the Corpus.")
        df = pd.DataFrame([self.meta])
        df.fillna("", inplace=True)
        return df

    @cached_property
    def num_active_tokens(self) -> int:
        """Return the number of active tokens in the Corpus."""
        if len(self.active_terms) == 0:
            return 0
        return sum(
            record.num_tokens()
            for record in self.records.values()
            if record.is_active and record.is_parsed
        )

    @cached_property
    @property
    def num_active_terms(self) -> int:
        """Return the number of active terms in the Corpus."""
        if len(self.active_terms) == 0:
            return 0
        return len(self.active_terms)

    def _add_to_corpus(self, record: Record, cache: Optional[bool] = False) -> None:
        """Add a record to the Corpus.

        Args:
            record (Record): A Record doc.
        """
        # Update corpus records table
        meta = record.model_dump(exclude=["content", "terms", "text", "tokens"])
        num_tokens = record.num_tokens() if record.is_parsed else 0
        num_terms = record.num_terms() if record.is_parsed else 0
        meta["num_tokens"] = num_tokens
        meta["num_terms"] = num_terms
        self.meta.append(meta)

        # Save the record to disk -- currently, this is always done
        corpus_dir = Path(self.corpus_dir)
        filename = f"{record.id}.bin"
        filepath = corpus_dir / "data" / filename
        record.meta["filename"] = filename
        record.meta["filepath"] = filepath
        record.to_disk(record, record.filename)

        # Update the Corpus records dictionary
        if cache:
            self.records[record.id] = record
        else:
            self.records[record.id] = None

        # Update the Corpus names
        self.names.append({record.name: record.id})

        # Update the Corpus statistics
        self._update_corpus_state()

    def _ensure_unique_name(self, name: str = None) -> str:
        """Ensure that no names are duplicated in the Corpus.

        Args:
            name (str): The record name.

        Returns:
            A string.
        """
        if not name:
            return f"untitled_{uuid.uuid1()}"
        if name in self.names:
            return f"{name}_{uuid.uuid1()}"
        return name

    def _generate_unique_id(self, type: str = "uuid4") -> str:
        """Generate a unique ID for the record.

        Args:
            type (str): The type of ID to generate. Can be "integer" or "uuid4". Defaults to "uuid4".

        Returns:
            str: A unique ID for the record.
        """
        if type == "integer":
            # Generate an integer ID
            return max(self.records.keys(), default=0) + 1
        elif type == "uuid4":
            # Generate initial UUID
            new_id = str(uuid.uuid4())

            # Keep generating new UUIDs until one is not in the records dic
            while new_id in self.records:
                new_id = str(uuid.uuid4())
            return new_id
        else:
            raise LexosException(
                f"Invalid ID type '{type}'. Must be 'integer' or 'uuid4'."
            )

    def _get_by_name(self, name: str) -> str:
        """Get a record ID from the Corpus by name.

        Args:
            name (str): The name of the record to get.

        Returns:
            str: The record ID.
        """
        if name not in self.names:
            raise LexosException(
                f"Record with name {name} does not exist in the Corpus."
            )
        return self.names[name]

    def _update_corpus_state(self):
        """Update the Corpus state after adding or removing records.

        Note:
            This method recalculates the number of documents, active documents,
            terms, tokens, and unique terms in the entire Corpus.
        """
        self.num_docs = len(self.records)
        self.num_active_docs = sum(1 for r in self.records if r.is_active)
        self.num_terms = sum(r.num_terms() for r in self.records if r.is_parsed)
        self.num_tokens = sum(r.num_tokens() for r in self.records if r.is_parsed)
        srsly.write_json(
            self.corpus_dir / self.corpus_metadata_file,
            self.model_dump_json(exclude=["content", "terms", "text", "tokens"]),
        )

    @validate_call(config=model_config)
    def add(
        self,
        content: Doc | Record | str | list[Doc | Record | str],
        name: Optional[str] = None,
        is_active: Optional[bool] = True,
        model: Optional[str] = None,
        extensions: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        id_type: Optional[str] = "uuid4",
        cache: Optional[bool] = False,
    ):
        """Add a document the Corpus.

        Args:
            content (Doc | Record | str | list[Doc | Record | str]): A text string, Record, or a spaCy document, or a list of any of these.
            name (str): A name for the document.
            is_active (bool): Whether or not the document is active.
            model (str): The name of the language model used to parse the document (optional).
            extensions (list[str]): A list of extension names to add to the document.
            metadata (dict[str, Any]): A dict containing any metadata.
            id_type (str): The type of ID to generate. Can be "integer" or "uuid4". Defaults to "uuid4".
            cache (bool): Whether or not to cache the record.
        """
        # Ensure that content is a list of Records, Docs, or strings
        if not isinstance(content, (Doc, Record, str, list)):
            content = [content]

        for item in content:
            # Generate a unique ID for the record
            new_id = self._generate_unique_id(type=id_type)

            # Keep generating new UUIDs until one is not in the records dic
            while new_id in self.records:
                new_id = str(uuid.uuid4())

            if isinstance(item, Record):
                record = item
                if record.id and record.id in self.records:
                    raise LexosException(
                        f"Record with ID {record.id} already exists in the Corpus."
                    )
                elif not record.id:
                    record.id = new_id
            else:
                record = Record(
                    id=new_id,
                    name=self._ensure_unique_name(name),
                    is_active=is_active,
                    content=item,
                    model=model,
                    extensions=extensions,
                    data_source=None,
                    meta=metadata,
                )

            # Add arbitrary metadata properties
            if metadata:
                record.metadata.update(metadata)

            # Add the record to the Corpus
            self._add_to_corpus(record, cache=cache)

    @validate_call(config=model_config)
    def get(
        self,
        id: Optional[str | list[str]] = None,
        name: Optional[str | list[str]] = None,
    ) -> Record | list[Record]:
        """Get a record from the Corpus by ID.

        Tries to get the record from memory; otherwise loads it from file.

        Args:
            id (str | list[str]): A document id or list of ids from the Corpus records.
            name (str | list[str]): A document name or list of names from the Corpus records.

        Returns:
            Record | list[Record]: The record(s) with the given ID(s) or name(s).
        """
        # Ensure either id or name is provided
        if not id and not name:
            raise LexosException(
                "Must provide either an ID or a name to remove a record."
            )

        # Ensure id is a list
        if isinstance(id, str):
            ids = [id]

        # If name is provided, get the ID from the name
        if name and not id:
            if isinstance(name, str):
                name = [name]
            ids = [self._get_by_name(n) for n in name]

        result = []
        for id in ids:
            # If the id is in the Corpus cache, return the record
            if id in self.records.keys():
                result.append(self.records[id])

            # Otherwise, load the record from file
            else:
                record = self.records[id]
                result.append(
                    record._from_disk(
                        record.meta["filepath"], record.model, self.model_cache
                    )
                )
        if len(result) == 1:
            return result[0]
        return result

    @validate_call(config=model_config)
    def get_stats(
        self,
        active_only: bool = True,
        type: str = "tokens",
        min_df: int | None = None,
        max_df: int | None = None,
        max_n_terms: int | None = None,
        token_list: list[tuple[str, str, list[str]]] = None,
    ) -> CorpusStats:
        """Get the Corpus statistics.

        Args:
            active_only (bool): If True, only include active records in the statistics. Defaults to True.
            type (str): The type of statistics to return. Can be "tokens" or "characters". Defaults to "tokens".
            min_df (int | None): Minimum document frequency for terms to be included in the statistics. Defaults to None.
            max_df (int | None): Maximum document frequency for terms to be included in the statistics. Defaults to None.
            max_n_terms (int | None): Maximum number of terms to include in the statistics. Defaults to None.
            token_list (list[tuple[str, str, list[str]]]): A list of tuples containing the record ID, name, and tokens. If not provided, it will be generated from the records.

        Returns:
            CorpusStats: An object containing the Corpus statistics.
        """

        def get_token_strings(record: Record) -> list[str]:
            """Get the token strings from a record.

            Args:
                record (Record): The Record object to get the token strings from.

            Returns:
                list[str]: A list of token strings from the record.
            """
            if record.is_parsed:
                return [token.text for token in record.content]
            # We could use xx_sent_ud_sm, but for now, split on whitespace
            else:
                return record.content.split()

        if not token_list:
            # Filter the records to only include active ones
            if active_only:
                records = [record for record in self.records if record.is_active]
            # Otherwise, include all records
            else:
                records = records

            # Get the token list from the records
            if type == "tokens":
                token_list = [
                    (record.id, record.name, get_token_strings(record))
                    for record in records
                ]
            elif type == "characters":
                token_list = [
                    (record.id, record.name, record.content.text)
                    if record.is_parsed
                    else (record.id, record.name, record.content)
                    for record in records
                ]

        return CorpusStats(
            docs=token_list, min_df=min_df, max_df=max_df, max_n_terms=max_n_terms
        )

    @validate_call(config=model_config)
    def load(
        self,
        path: Path | str = None,
        corpus_dir: Optional[Path | str] = None,
        cache: Optional[bool] = False,
    ) -> None:
        """Load a Corpus from a zip archive or directory.

        Args:
            path (Path | str): The path of the zip archive or directory to load.
            corpus_dir (Optional[Path | str]): The directory where the Corpus is to be unzipped.
            cache (Optional[bool]): Whether to cache the records in the Corpus. Defaults to False.
        """
        # Ensure that a corpus_dir exists, or create one if it doesn't
        if not corpus_dir:
            corpus_dir = Path(self.corpus_dir)
            corpus_dir.mkdir(parents=True, exist_ok=True)

        # If the path is a file, try to unpack it as a zip archive
        if Path(path).is_file():
            try:
                shutil.unpack_archive(path, corpus_dir)
            except shutil.ReadError as e:
                raise LexosException(
                    f"Failed to unpack archive: {e}. Ensure the file is a valid zip archive."
                )

        # Open the metadata file and load the metadata
        with open(corpus_dir / self.corpus_metadata_file, "r") as f:
            metadata = srsly.read_json(f)
            for key, value in metadata.items():
                setattr(self, key, value)

        # If cache is set, load the records into the model cache
        if cache:
            for record in self.records.values():
                if isinstance(record, Record):
                    # Load the record from disk
                    record.from_disk(
                        corpus_dir / "data" / f"{record.id}.bin",
                        model=record.model,
                        model_cache=self.model_cache,
                    )
                else:
                    raise LexosException(
                        "Records in the Corpus must be of type Record."
                    )

    @validate_call(config=model_config)
    def save(self, path: Path | str = None) -> None:
        """Save the Corpus as a zip archive.

        Args:
            path (Path | str): The path to save the Corpus to.
        """
        shutil.make_archive(path / f"{self.name}", "zip", self.corpus_dir)

    @validate_call(config=model_config)
    def remove(
        self,
        id: Optional[str | list[str]] = None,
        name: Optional[str | list[str]] = None,
    ) -> None:
        """Remove a record from the corpus by ID.

        Args:
            id (str | list[str]): The ID of the record to remove.
            name (str | list[str]): The name of the record to remove.
        """
        # Ensure either id or name is provided
        if not id and not name:
            raise LexosException(
                "Must provide either an ID or a name to remove a record."
            )

        # Ensure id is a list
        if isinstance(id, str):
            ids = [id]

        # If name is provided, get the ID from the name
        if name and not id:
            if isinstance(name, str):
                name = [name]
            ids = self._get_by_name(name)

        for id in ids:
            # Remove the entry from the records dictionary and names list
            try:
                entry = self.records.pop(id)
            except KeyError:
                raise LexosException(
                    f"Record with ID {id} does not exist in the Corpus."
                )
            try:
                self.names.pop(entry["name"])
            except KeyError:
                raise LexosException(
                    f"Record with name {entry['name']} does not exist in the Corpus."
                )

        # Update the Corpus state after removing the record
        self._update_corpus_state()

    @validate_call(config=model_config)
    def term_counts(
        self, n: Optional[int] = 10, most_common: Optional[bool] = True
    ) -> Counter:
        """Get a Counter with the most common Corpus term counts.

        Args:
            n (Optional[int]): The number of most common terms to return. Defaults to 10.
            most_common (Optional[bool]): If True, return the n most common terms; otherwise, return the n least common terms.

        Returns:
            A collections.Counter object containing the n most common term counts for all records in the Corpus.
        """
        # Count the terms in all records
        counter = Counter(n)
        for record in self.records.values():
            if record.is_parsed:
                counter.update(record.terms)

        # Optionally filter the results
        if most_common and n:
            return counter.most_common(n)
        elif not most_common and n:
            return counter.most_common()[: -n - 1 : -1]
        elif not most_common and n is None:
            return counter.most_common()[::]
        else:
            return counter

    @validate_call(config=model_config)
    def to_df(
        self, exclude: list[str] = ["content", "terms", "tokens"]
    ) -> pd.DataFrame:
        """Return a table of the Corpus records.

        Args:
            exclude (list[str]): A list of fields to exclude from the dataframe. If you wish to exclude metadata fields with the same name as model fields, you can use the prefix "metadata_" to avoid conflicts.

        Returns:
            pd.DataFrame: A dataframe representing the records in the Corpus.
        """
        rows = []
        for record in record in self.records.values():
            # Get model categories
            row = record.model_dump(exclude=exclude)

            # Add metadata categories
            metadata = row.pop("meta", {})
            for key, value in metadata.items():
                if key in row and f"metadata_{key}" not in exclude:
                    key = f"metadata_{key}"
                row[key] = value

            # Append the row to the rows list
            rows.append(row)

        # Create a DataFrame from the rows
        df = pd.DataFrame(rows, columns=list(rows[0].keys()))
        df.fillna("", inplace=True)
        return df
