# Lexos Topwords Module: Keyword and Top-Word Extraction

The lexos.topwords module provides tools for extracting significant keywords from documents and identifying statistically over-represented words in a corpus when compared to a background set of texts. It enables developers to perform comparative text analysis and keyword discovery using both algorithmic and statistical methods.

## Core Classes

### TextacyKeywords (__init__.py)

* This class extracts representative keywords from a single text document using algorithms from the textacy library. It is useful for quickly summarizing the key terms of a document.
* Backend: textacy.extract.keyterms.
* Methods: Supports standard keyword extraction algorithms via the method parameter, including `textrank` and `sgrank`.
* Customization: The `topn` parameter allows you to specify the number of top keywords to return.
* Output: Returns a dictionary containing a list of keywords, where each keyword is a dictionary with a term and a score. Example: `{'keywords': [{'term': str, 'score': float}]}`.

### ZTestTopwords (__init__.py)

This class identifies statistically distinguishing words in a target corpus when compared against a background corpus. It uses a Z-test to score words based on how significantly more frequent they are in the target texts.

* Backend: numpy for statistical calculations and spaCy (via the Lexos Tokenizer) for text processing.
* Method: A Z-test is used to compare the proportions of word frequencies between the target and background corpora.
* Preprocessing: Includes boolean options for `case_sensitive` analysis and for removing `stopwords`, `punct` (punctuation), and `digits`.
* Customization: The `topn` parameter controls how many of the most significant words are returned.
* Output: Returns a dictionary containing a list of the top words, where each word is a dictionary with a term and its z_score. Example: `{'topwords': [{'term': str, 'z_score': float}]}`.

## Prerequisites and Installation

To effectively use and contribute to this module, ensure you have the following installed:

* Python 3.8+

* Core Libraries:

```bash
pip install pydantic numpy spacy textacy
```
* spaCy Language Model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

All classes operate on raw text strings or lists of strings.

Here's a minimal example for TextacyKeywords:
```python
from lexos.topwords import TextacyKeywords

# Example text
doc = "Lexos is a powerful tool for text analysis. This analysis helps scholars find keywords."

# Extract top 3 keywords using the 'textrank' method
keyword_extractor = TextacyKeywords(text=doc, method="textrank", topn=3)
results = keyword_extractor()

print(results)
# Expected output: {'keywords': [{'term': 'text analysis', 'score': 0.45}, ...]}
```
Here's a minimal example for ZTestTopwords:
```python
from lexos.topwords import ZTestTopwords

# Example target and background documents
target_docs = ["We study machine learning in Python.", "Python is used for machine learning."]
background_docs = ["This is a general text about programming.", "Some developers prefer other languages."]

# Find the top 3 distinguishing words in the target docs
ztest_extractor = ZTestTopwords(target_texts=target_docs, background_texts=background_docs, topn=3)
results = ztest_extractor()

print(results)
# Expected output: {'topwords': [{'term': 'machine learning', 'z_score': 1.98}, ...]}
```

## Development and Testing

This module includes a test suite to ensure functionality and stability.

To run the test suite and generate a detailed coverage report (this requires uv for dependency management in the project root):
```bash
uv run pytest --cov=src/lexos/topwords --cov-report=html tests/topwords
```
