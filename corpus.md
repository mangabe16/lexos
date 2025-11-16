# Lexos Corpus Module User Guide

## Introduction

The Lexos Corpus module provides a powerful and flexible way to manage collections of text documents for digital humanities, linguistics, and text analysis projects. It is designed to make it easy to store, organize, filter, and process large corpora, whether you are working with plain text or parsed linguistic data (such as spaCy Docs).

The module supports multiple backends, including an in-memory store and a robust SQLite database for persistent storage. This guide focuses on the most common workflows and the recommended SQLite backend.

---

## Why Use the Corpus Module?

- **Organize and manage large collections of texts**
- **Store metadata and custom attributes for each document**
- **Filter, search, and retrieve records efficiently**
- **Support for parsed linguistic data (spaCy)**
- **Easy integration with other Lexos tools**

---

## Getting Started: The Corpus Model

A corpus is a collection of `Record` objects, each representing a document with its content and metadata. You can create records from plain text, parsed spaCy Docs, or other sources.

### Example: Creating a Corpus and Adding Records

```python
from lexos.corpus.record import Record
from lexos.corpus.sqlite.database import SQLiteBackend

# Create a new SQLite-backed corpus (in-memory for demo)
backend = SQLiteBackend(database_path=":memory:")

# Add a plain text record
record1 = Record(
    id="doc1",
    name="First Document",
    content="This is the first document.",
    model=None
)
backend.add_record(record1)

# Add a parsed spaCy Doc (requires spaCy)
import spacy
nlp = spacy.blank("en")
doc = nlp("This is a parsed document.")
record2 = Record(
    id="doc2",
    name="Second Document",
    content=doc,
    model="en_core_web_sm"
)
backend.add_record(record2)

---

The corpus module provides a variety of summary statistics to help you understand and analyze your collection. These statistics are especially useful for exploring corpus structure, linguistic diversity, and data quality before deeper analysis.

### What Statistics Can You Generate?

- **Total Records:** The number of documents in your corpus.
- **Active Records:** How many records are marked as active (useful for filtering).
- **Parsed Records:** Number of records that have been processed with spaCy or another NLP tool.
- **Total Tokens:** The total number of tokens (words) across all documents.
- **Average Tokens per Record:** The mean token count per document.
- **Vocabulary Size:** The number of unique terms in the corpus.
- **Hapax Legomena Ratio:** Proportion of words that occur only once.
- **Dislegomena Ratio:** Proportion of words that occur exactly twice.
- **Lexical Diversity:** Measures such as Type-Token Ratio (TTR), Corrected TTR (CTTR), and Root TTR (RTTR).
- **Document Length Distribution:** Minimum, maximum, mean, and standard deviation of document lengths.
- **Metadata Coverage:** How many records have specific metadata fields populated.
- **Model Coverage:** Distribution of NLP models used for parsing (if applicable).

### How to Access Statistics

You can retrieve corpus statistics using the backend’s `get_corpus_stats()` method:

```python
stats = backend.get_corpus_stats()
print(stats)
```

**Example Output:**
```python
{
    'total_records': 2,
    'active_records': 2,
    'parsed_records': 1,
    'total_tokens': 10,
    'avg_tokens_per_record': 5.0,
    'vocabulary_size': 8,
    'hapax_ratio': 0.25,
    'dislegomena_ratio': 0.10,
    'lexical_diversity': {
        'ttr': 0.8,
        'cttr': 0.7,
        'rttr': 0.75
    },
    'doc_length': {
        'min': 4,
        'max': 6,
        'mean': 5.0,
        'std': 1.0
    },
    'metadata_coverage': {
        'author': 2,
        'year': 2
    },
    'model_coverage': {
        'en_core_web_sm': 1,
        'None': 1
    }
}
```

### Technical Notes

- **Custom Statistics:** You can extend or customize statistics by exporting the corpus to a DataFrame and using pandas or other libraries.
- **Parsed Records:** Some statistics (like lexical diversity) are more meaningful if your records are parsed with an NLP model.
- **Metadata Coverage:** If you use custom metadata fields, statistics will show how many records have each field.

### Visualizing Statistics

For deeper exploration, export your statistics or corpus data to pandas and use visualization libraries:

```python
import pandas as pd
df = backend.to_df()
df['token_count'].hist(bins=20)
```

---

Corpus statistics are a great starting point for understanding your data, identifying outliers, and planning further analysis. They help ensure your corpus is balanced, diverse, and ready for research!

# Get records parsed with spaCy
parsed_records = backend.filter_records(is_parsed=True)

# Filter by token count
long_records = backend.filter_records(min_tokens=100)
```

---

## Searching the Corpus

The SQLite backend supports full-text search using FTS5:

```python
# Search for records containing a keyword
results = backend.search_records("keyword")
for record in results:
    print(record.name, record.content)
```

---

## Accessing Corpus Statistics

You can quickly get summary statistics for your corpus:

```python
stats = backend.get_corpus_stats()
print(stats)
# Example output:
# {'total_records': 2, 'active_records': 2, 'parsed_records': 1, 'total_tokens': 10, ...}
```

---

## Technical Notes

- **IDs:** Each record should have a unique `id` (string or UUID).
- **Models:** If you use spaCy Docs, specify the model name (e.g., `"en_core_web_sm"`).
- **Persistence:** Use a file path for `database_path` to save your corpus between sessions.
- **Metadata:** You can attach custom metadata to each record for advanced filtering.

---

## Closing and Saving

Always close the backend when done to ensure data is saved:

```python
backend.close()
```

---

## Next Steps

- Explore advanced filtering and metadata features
- Integrate with Lexos analysis and visualization tools
- See the Lexos documentation for more details and examples

---

Lexos Corpus makes it easy to manage and analyze your texts, whether you're working with a handful of documents or thousands. Happy text mining!
