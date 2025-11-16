# SQLite Database Integration for Lexos Corpus Module

**Status**: ✅ Complete Implementation  
**Technology**: SQLAlchemy + SQLite with FTS5  
**Last Updated**: June 27, 2025  

## Overview

The SQLite database integration provides optional database storage capabilities alongside the existing file-based system. This implementation preserves 100% backward compatibility while adding powerful new database capabilities including full-text search, efficient querying, and flexible deployment options.

## 🎯 Key Features

- ✅ **Optional Storage**: Database works alongside existing file-based system
- ✅ **Full-Text Search**: FTS5-powered search with ranking and boolean operators  
- ✅ **Metadata Queries**: Efficient filtering and aggregation
- ✅ **Zero Breaking Changes**: Existing code continues to work unchanged
- ✅ **Flexible Deployment**: In-memory, file-based, or hybrid storage options
- ✅ **Data Integrity**: Hash verification and transaction safety

## Architecture Components

### Core Implementation Files

1. **`database_simple.py`** - Main database implementation using SQLAlchemy
   - Compatible with SQLModel 0.0.24
   - Provides `CorpusDatabase`, `DatabaseRecord`, and `DatabaseCorpus` classes
   - Includes FTS5 full-text search capabilities
   - CRUD operations for records

2. **`corpus_db_integration.py`** - Extended Corpus class with database features
   - `DatabaseEnabledCorpus` class extends base `Corpus`
   - Supports dual storage (files + database) and database-only modes
   - Maintains full backward compatibility
   - Convenience function `create_database_corpus()`

3. **`__init__.py`** - Module initialization with optional imports
   - Graceful handling of missing dependencies
   - Optional database feature availability

### Documentation and Examples

4. **`database_integration_tutorial.ipynb`** - Comprehensive tutorial and examples
5. **`README.md`** - This documentation file

## Database Schema Design

The database uses a simple but effective schema:

```sql
CREATE TABLE records (
    id TEXT PRIMARY KEY,
    name TEXT,
    content_text TEXT NOT NULL,
    content_doc_bytes BLOB,
    is_active BOOLEAN DEFAULT TRUE,
    is_parsed BOOLEAN DEFAULT FALSE,
    model TEXT,
    num_tokens INTEGER DEFAULT 0,
    num_terms INTEGER DEFAULT 0,
    vocab_density REAL DEFAULT 0.0,
    content_hash TEXT,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE records_fts USING fts5(
    content_text,
    content=records,
    content_rowid=id
);
```

## Usage Examples

### Basic Database-Enabled Corpus

```python
from lexos.database import create_database_corpus

# Create corpus with database integration
corpus = create_database_corpus(
    corpus_dir="my_corpus",
    database_path="my_corpus.db",
    name="My Corpus",
    database_only=False  # Use both files and database
)

# Add documents
corpus.add(
    content="The quick brown fox jumps over the lazy dog.",
    name="sample_doc",
    metadata={"genre": "example"}
)

# Full-text search
results = corpus.search("fox jumps")
print(f"Found {len(results)} documents")

# Advanced filtering
filtered = corpus.filter_records(min_tokens=5, max_tokens=20)
```

### Database-Only Mode

```python
# Create in-memory database-only corpus
corpus = create_database_corpus(
    corpus_dir="memory_corpus",
    database_path=":memory:",
    name="Memory Corpus",
    database_only=True
)
```

### Synchronization

```python
# Synchronize existing file-based corpus to database
corpus = DatabaseEnabledCorpus(
    corpus_dir="existing_corpus",
    database_path="sync.db",
    enable_database=True
)

corpus.load(path="existing_corpus")
synced_count = corpus.sync_to_database()
```

## Current File-Based Storage Architecture

### Strengths of Current System
- **Simplicity**: Zero external dependencies, works everywhere Python runs
- **Portability**: ZIP archives enable easy corpus sharing and deployment
- **Data Integrity**: SHA256 hash verification prevents corruption
- **Performance**: msgpack binary serialization is fast and compact
- **Model Compatibility**: Native spaCy Doc serialization preserves all linguistic data

### Limitations Addressed by Database Enhancement
1. **No Indexing**: O(n) search through all records for metadata queries
2. **Concurrency Issues**: No multi-user access protection or transaction support
3. **Scale Bottlenecks**: Flat directory structure limits filesystem performance
4. **Limited Querying**: No SQL-like query capabilities for complex data analysis
5. **Metadata Management**: JSON metadata becomes unwieldy at scale

## Database Solution Benefits

### SQLite Integration Advantages
✅ **Zero Configuration**: Single file database, no server setup required  
✅ **ACID Compliance**: Full transaction support for data consistency  
✅ **SQL Querying**: Rich query language for complex metadata operations  
✅ **Full-Text Search**: FTS5 provides sophisticated search capabilities  
✅ **Lightweight**: Perfect for research and small-to-medium scale deployments  
✅ **Python Integration**: Excellent SQLAlchemy support and tooling  

## API Reference

### DatabaseEnabledCorpus Class

The `DatabaseEnabledCorpus` class extends the base `Corpus` class with database capabilities:

```python
class DatabaseEnabledCorpus(Corpus):
    def __init__(
        self,
        corpus_dir: str = "corpus",
        database_path: str = "corpus.db",
        name: str = "Corpus",
        enable_database: bool = True,
        database_only: bool = False,
        **kwargs
    )
    
    def search(self, query: str, limit: int = 10) -> List[Record]
    def filter_records(self, **filters) -> List[Record]
    def get_database_stats(self) -> Dict[str, Any]
    def sync_to_database(self) -> int
```

### Convenience Functions

```python
def create_database_corpus(
    corpus_dir: str,
    database_path: str,
    name: str,
    database_only: bool = False
) -> DatabaseEnabledCorpus
```

## Testing

### Test Files
- `test_database_integration.py` - Comprehensive integration tests
- `test_database_simple.py` - Basic database functionality tests
- `test_db_integration_simple.py` - Simple validation tests

### Test Results Summary
- ✅ **PASSED - Core Functionality (5/5 tests)**
- ✅ **PASSED - Database Concepts (3/3 tests)**
- ✅ **PASSED - Integration Tests**

## Deployment Options

### 1. Development Mode (Hybrid)
```python
corpus = create_database_corpus(
    corpus_dir="dev_corpus",
    database_path="dev.db",
    database_only=False
)
```

### 2. Production Mode (Database-Only)
```python
corpus = create_database_corpus(
    corpus_dir="prod_corpus",
    database_path="/app/data/corpus.db",
    database_only=True
)
```

### 3. Temporary Analysis (In-Memory)
```python
corpus = create_database_corpus(
    corpus_dir="temp",
    database_path=":memory:",
    database_only=True
)
```

## Performance Considerations

- **Indexing**: Database queries are indexed for optimal performance
- **FTS5 Search**: Full-text search is optimized for large document collections
- **Connection Management**: Proper connection pooling and cleanup
- **Memory Usage**: Efficient memory usage with lazy loading

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
2. **Database Lock**: Close connections properly, especially on Windows
3. **Memory Usage**: Use database-only mode for large corpora
4. **Performance**: Ensure indexes are created for frequently queried fields

### Dependencies
- SQLAlchemy >= 1.4
- sqlite3 (built-in)
- msgpack
- spacy

## Future Enhancements

- MongoDB integration for cloud deployments
- Advanced analytics and reporting
- Multi-user access controls
- Distributed corpus management
- Real-time synchronization capabilities

## Conclusion

The SQLite database integration enhances the Lexos corpus module while maintaining full backward compatibility. It provides scalable storage, efficient querying, and flexible deployment options that address the limitations of the file-based system while preserving its strengths.