"""corpus_db_integration.py.

Database integration layer for the Corpus class.

This module extends the existing Corpus class with optional SQLite database
capabilities while maintaining full compatibility with the file-based system.

Last Updated: July 14, 2025
Last Tested: July 10, 2025

"""

import json
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import Field, validate_call

from lexos.corpus.corpus import Corpus
from lexos.corpus.record import Record

# Import database components - use simple implementation
from lexos.database.database_simple import (
    CorpusDatabase,
    DatabaseCorpus,
    DatabaseRecord,
)
from lexos.exceptions import LexosException


class DatabaseEnabledCorpus(Corpus):
    """Extended Corpus class with optional SQLite database integration.

    This class extends the standard Corpus with database capabilities:
    - Dual storage: files + database
    - Full-text search across records
    - Efficient metadata queries
    - Optional database-only mode

    The database integration is completely optional and does not break
    existing file-based workflows.
    """

    # Add database-related fields to the Pydantic model
    enable_database: bool = Field(
        default=False, description="Whether to enable database storage"
    )
    database_only: bool = Field(
        default=False, description="Whether to use database-only mode"
    )
    database_path: Optional[str] = Field(
        default=None, description="Path to SQLite database file"
    )
    db: Optional[Any] = Field(
        default=None, description="Database connection object", exclude=True
    )

    def __init__(self, **data):
        """Initialize corpus with optional database integration.

        Args:
            database_path: Path to SQLite database file (None for memory-based)
            enable_database: Whether to enable database storage alongside files
            database_only: Whether to use database-only mode (no file storage)
            **data: Standard Corpus initialization parameters
        """
        # Extract database-specific parameters
        database_path = data.pop("database_path", None)
        enable_database = data.pop("enable_database", False)
        database_only = data.pop("database_only", False)

        # Set the database fields
        data["enable_database"] = enable_database
        data["database_only"] = database_only
        data["database_path"] = database_path

        # Initialize parent class
        super().__init__(**data)

        # Initialize database if enabled
        if self.enable_database or self.database_only:
            db_path = database_path or f"{self.corpus_dir}/corpus.db"
            self.db = CorpusDatabase(database_path=db_path)
            self._initialize_corpus_metadata()
        else:
            self.db = None

    def _add_database_only(
        self,
        content,
        name: Optional[str] = None,
        is_active: Optional[bool] = True,
        model: Optional[str] = None,
        extensions: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        id_type: Optional[str] = "uuid4",
    ):
        """Add records in database-only mode without file storage."""
        from spacy.tokens import Doc

        # Handle single or multiple content items
        if isinstance(content, (Doc, Record, str)):
            items = [content]
        else:
            items = list(content)

        for item in items:
            # Generate unique ID
            new_id = self._generate_unique_id(type=id_type)

            if isinstance(item, Record):
                record = item
            else:
                record_kwargs = dict(
                    id=new_id,
                    name=self._ensure_unique_name(name),
                    is_active=is_active,
                    content=item,
                    model=model,
                    data_source=None,
                )
                if extensions is not None:
                    record_kwargs["extensions"] = extensions
                if metadata is not None:
                    record_kwargs["meta"] = metadata
                record = Record(**record_kwargs)

                # Note: Records are created with string content and can be parsed later if needed
                # The database stores both parsed and unparsed content efficiently

            # Add to in-memory records
            record_id_str = str(record.id)
            self.records[record_id_str] = record
            self.names[record.name] = record_id_str

            # Store in database
            if self.db:
                self.db.add_record(record)

        # Update corpus state
        self._update_corpus_state()

    def __del__(self):
        """Destructor to ensure database connections are closed."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _initialize_corpus_metadata(self):
        """Initialize corpus metadata in the database."""
        if not self.db:
            return

        with self.db.SessionLocal() as session:
            # Check if corpus metadata exists
            corpus_id = self.name or "default"
            existing = (
                session.query(DatabaseCorpus)
                .filter(DatabaseCorpus.corpus_id == corpus_id)
                .first()
            )

            if not existing:
                # Create new corpus metadata
                corpus_metadata = DatabaseCorpus()
                corpus_metadata.corpus_id = corpus_id
                corpus_metadata.name = self.name
                corpus_metadata.num_docs = self.num_docs
                corpus_metadata.num_active_docs = self.num_active_docs
                corpus_metadata.num_tokens = self.num_tokens
                corpus_metadata.num_terms = self.num_terms
                corpus_metadata.corpus_dir = self.corpus_dir
                corpus_metadata.metadata_json = json.dumps(self.meta, default=str)
                corpus_metadata.analysis_results_json = json.dumps(
                    self.analysis_results, default=str
                )
                corpus_metadata.corpus_fingerprint = self._generate_corpus_fingerprint()
                corpus_metadata.created_at = self._get_timestamp()
                corpus_metadata.updated_at = self._get_timestamp()
                session.add(corpus_metadata)
                session.commit()

    def _update_corpus_state(self):
        """Update corpus state in both memory and database."""
        # Update in-memory state
        super()._update_corpus_state()

        # Update database metadata if enabled
        if self.db:
            with self.db.SessionLocal() as session:
                corpus_id = self.name or "default"
                corpus_metadata = (
                    session.query(DatabaseCorpus)
                    .filter(DatabaseCorpus.corpus_id == corpus_id)
                    .first()
                )

                if corpus_metadata:
                    corpus_metadata.num_docs = self.num_docs
                    corpus_metadata.num_active_docs = self.num_active_docs
                    corpus_metadata.num_tokens = self.num_tokens
                    corpus_metadata.num_terms = self.num_terms
                    corpus_metadata.metadata_json = json.dumps(self.meta, default=str)
                    corpus_metadata.analysis_results_json = json.dumps(
                        self.analysis_results, default=str
                    )
                    corpus_metadata.corpus_fingerprint = (
                        self._generate_corpus_fingerprint()
                    )
                    corpus_metadata.updated_at = self._get_timestamp()

                    session.commit()

    @validate_call
    def add(
        self,
        content,
        name: Optional[str] = None,
        is_active: Optional[bool] = True,
        model: Optional[str] = None,
        extensions: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        id_type: Optional[str] = "uuid4",
        cache: Optional[bool] = False,
        store_in_db: Optional[bool] = None,
    ):
        """Add a record to the corpus with optional database storage.

        Args:
            store_in_db: Override database storage for this record (None = use corpus setting)
            **kwargs: Standard Corpus.add() parameters
        """
        # Determine storage strategy
        use_db = (
            store_in_db
            if store_in_db is not None
            else self.enable_database or self.database_only
        )
        use_files = not self.database_only

        # Get current record count to track new additions
        initial_record_count = len(self.records)

        # Add using parent implementation if using files
        if use_files:
            super().add(
                content=content,
                name=name,
                is_active=is_active,
                model=model,
                extensions=extensions,
                metadata=metadata,
                id_type=id_type,
                cache=cache,
            )
        else:
            # Database-only mode - implement add logic without file storage
            self._add_database_only(
                content=content,
                name=name,
                is_active=is_active,
                model=model,
                extensions=extensions,
                metadata=metadata,
                id_type=id_type,
            )

        # Also store in database if enabled and we're using file storage
        if use_db and self.db and use_files:
            # Get the newly added records
            current_records = list(self.records.values())
            new_records = current_records[initial_record_count:]

            for record in new_records:
                try:
                    # Note: Records can be parsed later if needed
                    # The database efficiently stores both parsed and unparsed content

                    self.db.add_record(record)
                except Exception as e:
                    # Log error but don't fail the entire operation
                    print(f"Warning: Failed to add record {record.id} to database: {e}")

    @validate_call
    def filter_records(
        self,
        is_active: Optional[bool] = None,
        is_parsed: Optional[bool] = None,
        model: Optional[str] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        limit: Optional[int] = None,
        use_database: bool = True,
    ) -> list[Record]:
        """Filter records by various criteria.

        Args:
            is_active: Filter by active status
            is_parsed: Filter by parsed status
            model: Filter by spaCy model name
            min_tokens: Minimum number of tokens
            max_tokens: Maximum number of tokens
            limit: Maximum number of results
            use_database: Whether to use database filtering (vs in-memory)

        Returns:
            List of matching Record objects
        """
        if use_database and self.db:
            return self.db.filter_records(
                is_active=is_active,
                is_parsed=is_parsed,
                model=model,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                limit=limit,
            )
        else:
            # Fallback to in-memory filtering
            filtered_records = []
            for record in self.records.values():
                if is_active is not None and record.is_active != is_active:
                    continue
                if is_parsed is not None and record.is_parsed != is_parsed:
                    continue
                if model is not None and record.model != model:
                    continue
                if min_tokens is not None:
                    try:
                        if record.num_tokens() < min_tokens:
                            continue
                    except:
                        continue
                if max_tokens is not None:
                    try:
                        if record.num_tokens() > max_tokens:
                            continue
                    except:
                        continue

                filtered_records.append(record)

                if limit and len(filtered_records) >= limit:
                    break

            return filtered_records

    @validate_call
    def get_database_stats(self) -> dict[str, Any]:
        """Get corpus statistics from the database.

        Returns:
            Dictionary containing database-derived statistics

        Raises:
            LexosException: If database is not enabled
        """
        if not self.db:
            raise LexosException(
                "Database is not enabled. Initialize corpus with enable_database=True."
            )

        return self.db.get_corpus_stats()

    @validate_call
    def search(
        self,
        query: str,
        limit: int = 100,
        include_inactive: bool = False,
        model_filter: Optional[str] = None,
        load_from_db: bool = True,
    ) -> list[Record]:
        """Perform full-text search on corpus records.

        Args:
            query: FTS5 search query string
            limit: Maximum number of results to return
            include_inactive: Whether to include inactive records
            model_filter: Optional filter by spaCy model name
            load_from_db: Whether to load results from database (vs memory)

        Returns:
            List of matching Record objects

        Raises:
            LexosException: If database is not enabled
        """
        if not self.db:
            raise LexosException(
                "Database is not enabled. Initialize corpus with enable_database=True to use search."
            )

        return self.db.search_records(
            query=query,
            limit=limit,
            include_inactive=include_inactive,
            model_filter=model_filter,
        )

    @validate_call
    def sync_to_database(self, overwrite: bool = False) -> int:
        """Synchronize existing file-based records to the database.

        Args:
            overwrite: Whether to overwrite existing database records

        Returns:
            Number of records synchronized

        Raises:
            LexosException: If database is not enabled
        """
        if not self.db:
            raise LexosException(
                "Database is not enabled. Initialize corpus with enable_database=True."
            )

        synced_count = 0

        for record in self.records.values():
            try:
                if overwrite:
                    # Check if exists and update
                    existing = self.db.get_record(str(record.id), include_doc=False)
                    if existing:
                        self.db.update_record(record)
                    else:
                        self.db.add_record(record)
                else:
                    # Only add if doesn't exist
                    existing = self.db.get_record(str(record.id), include_doc=False)
                    if not existing:
                        self.db.add_record(record)

                synced_count += 1

            except Exception as e:
                # Log error but continue with other records
                print(f"Warning: Failed to sync record {record.id}: {str(e)}")

        return synced_count

    @validate_call
    def load_from_database(
        self, include_docs: bool = False, active_only: bool = True
    ) -> int:
        """Load records from database into memory.

        Args:
            include_docs: Whether to deserialize spaCy Doc content
            active_only: Whether to load only active records

        Returns:
            Number of records loaded

        Raises:
            LexosException: If database is not enabled
        """
        if not self.db:
            raise LexosException(
                "Database is not enabled. Initialize corpus with enable_database=True."
            )

        # Clear existing records if loading from database
        self.records.clear()
        self.names.clear()

        # Load records from database
        filters = {"is_active": True} if active_only else {}
        db_records = self.db.filter_records(**filters)

        loaded_count = 0
        for record in db_records:
            # Add to in-memory structures
            record_id_str = str(record.id)
            self.records[record_id_str] = record
            self.names[record.name] = record_id_str
            loaded_count += 1

        # Update corpus state
        self._update_corpus_state()

        return loaded_count

    def close(self):
        """Close database connections and clean up resources."""
        if self.db:
            self.db.close()


# Convenience function to create database-enabled corpus
def create_database_corpus(
    corpus_dir: str = "corpus",
    database_path: Optional[Union[str, Path]] = None,
    name: Optional[str] = None,
    database_only: bool = False,
    **kwargs,
) -> DatabaseEnabledCorpus:
    """Create a database-enabled corpus with sensible defaults.

    Args:
        corpus_dir: Directory for file-based storage
        database_path: Path to SQLite database (None for auto-generated)
        name: Corpus name
        database_only: Whether to use database-only mode
        **kwargs: Additional Corpus initialization parameters

    Returns:
        DatabaseEnabledCorpus instance
    """
    if database_path is None:
        database_path = f"{corpus_dir}/corpus.db"

    return DatabaseEnabledCorpus(
        corpus_dir=corpus_dir,
        name=name,
        database_path=database_path,
        enable_database=True,
        database_only=database_only,
        **kwargs,
    )
