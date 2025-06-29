# Lexos Topwords Module

The lexos.topwords module provides tools for extracting significant keywords from documents and identifying statistically over-represented words in a corpus when compared to a background set of texts. It enables developers to perform comparative text analysis and keyword discovery using both algorithmic and statistical methods.

## Module Structure

This module is separated under 4 core files:

- `__init__.py`: Provides the base class for the module and its plugin classes, providing a common API.
- `keyterms.py`: Contains the `KeyTerms` class for keyterm extraction using Textacy's `keyterms.textrank()` and related algorithms.
- `ZTest.py`: Contains the `ZTest` class for topword extraction via Z score calculation.
- `comparison_handler.py`: Contains the `ComparisonHandler` class for comparison between documents and classes of documents within the corpus

## Difference on `keyterms` and `topwords`

The `KeyTerms` class uses Textacy's algorithmic keyword extraction to return the statistically distinguishing terms in a document, whereas the `ZTest` class uses a Z score to calculate which terms best distinguish a list of target documents in contrast to a list of background documents.

| Feature            | `keyterms` (KeyTerms)                                         | `topwords` (ZTest)                                         |
|--------------------|---------------------------------------------------------------|------------------------------------------------------------|
| **Purpose**        | Extracts representative or important terms from a single text  | Identifies words statistically over-represented in target texts compared to a background corpus |
| **Method**         | Algorithmic keyword extraction (e.g., TextRank, SGRank)       | Statistical comparison using Z-test                        |
| **Input**          | One document (string or spaCy Doc)                            | List of target documents and list of background documents  |
| **Best for**       | Summarizing or highlighting main topics in a single document  | Finding distinguishing words that set a group of texts apart from others |
| **Customization**  | Choice of extraction algorithm, number of keywords, ngram range | Preprocessing options (case, stopwords, etc.), ngram range, number of top words |
| **Output**         | Keyterms ranked by importance                                 | Words ranked by statistical significance (Z-score)         |
| **When to use**    | When you want to summarize or tag a document with key terms   | When you want to compare groups of texts and find what makes one group unique |

## Plugin Architecture

All topwords-related classes inherit from a common base class, `TopwordsPlugin`, which provides a consistent API (such as `to_dict()`, `to_df()`, and `to_list()`) for serialization and output. This makes it easy to add new analysis methods as plugins.

## spaCy Doc Extensions

Custom attributes are registered on spaCy `Doc` objects to store analysis results:

- `topwords`: Stores the top distinguishing words for a document.
- `keyterms`: Stores extracted keyterms for a document.

These attributes are set automatically by the analysis classes if you provide spaCy `Doc` objects.

## Classes

### `KeyTerms`

- Extracts representative keyterms from a single text document using algorithms from the textacy library.
- Backend: `textacy.extract.keyterms`.
- Methods: Supports standard keyword extraction algorithms via the method parameter, including `textrank` and `sgrank`.
- Customization: The `topn` parameter allows you to specify the number of top keyterms to return. The `ngrams` parameter allows you to control the ngram range for keyterm extraction.
- Output: After calling the instance, use `.to_dict()`, `.to_df()`, or `.to_list()` to get the results in your preferred format.

__Parameters:__

- `document`: The raw text or spaCy Doc to analyze.
- `method`: `"textrank"` or `"sgrank"` (default: `"textrank"`).
- `topn`: Number of top keyterms to return (default: 10).
- `ngrams`: Tuple specifying the ngram range, e.g., `(1, 3)` (default: `(1, 3)`).
- `model`: spaCy model name to use for tokenization (default: `"xx_sent_ud_sm"`).
- `tokenizer`: (optional) A `Tokenizer` instance.

__Usage:__

```python
from lexos.topwords.keyterms import KeyTerms

kw = KeyTerms(document="Your text here", method="textrank", topn=5)
kw()  # Run the analysis
print(kw.to_dict())  # List of dicts
print(kw.to_df())    # DataFrame
print(kw.to_list())  # List of tuples
# The keyterms are also set on the spaCy Doc as kw.document._.keyterms if you passed a Doc as input
```

### `ZTest`

- Identifies the most distinguishing words in a set of target documents compared to a background corpus using a Z-test for statistical significance.
- Backend: numpy for statistical calculations and spaCy (via the Lexos Tokenizer) for text processing.
- Method: A Z-test is used to compare the proportions of word frequencies between the target and background corpora.
- Preprocessing: Includes boolean options for `case_sensitive` analysis and for removing `stopwords`, `punct` (punctuation), and `digits`.
- Customization: The `topn` parameter controls how many of the most significant words are returned. The `ngrams` parameter allows you to control the ngram range for analysis.
- Output: After calling the instance, use `.to_dict()`, `.to_df()`, or `.to_list()` to get the results in your preferred format.

__Parameters:__

- `target_documents`: List of target documents (strings or spaCy Docs).
- `background_documents`: List of background documents (strings or spaCy Docs).
- `topn`: Number of top words to return (default: 10).
- `ngrams`: Tuple specifying the ngram range, e.g., `(1, 1)` for unigrams only (default: `(1, 1)`).
- `case_sensitive`, `remove_stopwords`, `remove_punct`, `remove_digits`: Preprocessing options.
- `model`: spaCy model name to use for tokenization (default: `"xx_sent_ud_sm"`).
- `tokenizer`: (optional) A `Tokenizer` instance.
- `docs`: (optional) List of spaCy `Doc` objects to set results on.
- `output_format`: Output format: `"dict"`, `"dataframe"`, `"list_of_dicts"`, or `"list_of_tuples"` (default: `"dict"`).

__Usage:__

```python
from lexos.topwords.ZTest import ZTest
import spacy

nlp = spacy.blank("en")
target_documents = ["This is a test.", "Another test document."]
background_documents = ["Background text here.", "More background docs."]
docs = [nlp(text) for text in target_documents]

ztest = ZTest(
    target_documents=target_documents,
    background_documents=background_documents,
    docs=docs,
    ngrams=(1, 2)  # Example: use unigrams and bigrams
)
ztest()  # Run the analysis
print(ztest.to_dict())  # List of dicts
print(ztest.to_df())    # DataFrame
print(ztest.to_list())  # List of tuples
# Each doc in docs now has doc._.topwords set to the top words
```

## ZTestComparison

The `ZTestComparison` class provides advanced comparative analysis between documents or groups of documents (categories/classes) using Z-score statistics. It enables you to compare individual documents, categories, or all-vs-all groupings to identify distinguishing words.

### Methods

| Method Name                                              | Description                                                                                                   |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `model_post_init(self, __context)`                       | Validates and initializes labels and categories after model initialization.                                   |
| `compare_docs_to_corpus(self, docs, background_cats=None, **kwargs)` | Compares each document in `docs` to the rest of the corpus (optionally filtered by background categories).    |
| `compare_cat_to_corpus(self, cat, background_cats=None, **kwargs)`   | Compares all documents in a given category to documents in specified background categories.                   |
| `compare_all_cats_to_corpus(self, background_cats=None, **kwargs)`   | Compares each category to all other categories (or specified background categories).                          |
| `compare_each_doc_to_other_classes(self, **kwargs)`                  | For each class, compares each document in that class to all documents in other classes.                       |
| `compare_each_class_to_each_other_class(self, **kwargs)`             | For each class, compares it to every other class individually, returning a nested dictionary.                 |
| `_get_doc_labels(self, docs)`                            | Helper: Gets labels for specific documents.                                                                  |
| `_create_doc_mapping(self)`                              | Helper: Creates a mapping from document content to corpus indices.                                            |
| `_determine_background_categories(self, cat, background_cats, available_cats)` | Helper: Determines which categories to use as background.                                                     |
| `_separate_docs_by_category(self, target_cat, background_categories)`| Helper: Separates documents into target and background groups by category.                                    |
| `_format_output(self, results)`                          | Helper: Formats the output according to the specified output format.                                          |
| `to_df(results)` *(static)*                              | Converts results to a pandas DataFrame with columns: label, term, z_score.                                   |
| `to_list_of_dicts(results)` *(static)*                   | Converts results to a list of dictionaries.                                                                  |
| `to_list_of_tuples(results)` *(static)*                  | Converts results to a list of tuples.                                                                        |

### Example Usage

```python
from lexos.topwords.ZTest import ZTestComparison

# Suppose you have a list of documents and their categories
docs = ["text of doc1", "text of doc2", "text of doc3"]
categories = ["A", "B", "A"]

zcomp = ZTestComparison(documents=docs, categories=categories, topn=5)
results = zcomp.compare_all_cats_to_corpus()
df = ZTestComparison.to_df(results)
print(df)
```

You can use the various comparison methods to analyze differences between documents or groups, and output results as a DataFrame, list of dicts, or list of tuples using the static methods.

## Comparison Handler

The `ComparisonHandler` class provides utilities for comparative analysis, allowing you to compare documents or groups of documents in various ways. This is useful for workflows where you want to:

- Compare each document to the rest of the corpus
- Compare each document in a class to all documents in other classes
- Compare each class (group of documents) to all other classes

__Parameters__:

- `cls`: The comparison class to use for analysis (e.g., `ZTest`, `KeyTerms`). This class should accept `target_documents`, `background_documents`, and any additional keyword arguments.
- `**kwargs`: Additional keyword arguments to pass to the comparison class (such as `topn`, `ngrams`, preprocessing options, etc.).
- For the comparison methods:
    - `compare_each_doc_to_corpus(documents: list[str])`: Takes a list of documents (strings or spaCy Docs) and compares each document to the rest of the corpus.
    - `compare_each_doc_to_other_classes(class_docs: dict[str, list[str]])`: Takes a dictionary mapping class names to lists of documents, and compares each document in a class to all documents in other classes.
    - `compare_each_class_to_other_classes(class_docs: dict[str, list[str]])`: Takes a dictionary mapping class names to lists of documents, and compares each class (group of documents) to all documents in other classes.
- Returns results as lists or dictionaries of results from the comparison class.

### Usage

```python
from lexos.topwords.comparison_handler import ComparisonHandler
from lexos.topwords.ZTest import ZTest

# Example: Compare each document to the rest of the corpus
docs = ["doc1 text", "doc2 text", "doc3 text"]
handler = ComparisonHandler(ZTest, topn=5)
results = handler.compare_each_doc_to_corpus(docs)
```

## Extending

To add a new analysis method, inherit from `TopwordsPlugin` and implement your logic. You can also register new custom spaCy Doc extensions as needed.
