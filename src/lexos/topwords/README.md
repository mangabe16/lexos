# Lexos Topwords Module

The lexos.topwords module provides tools for extracting significant keywords from documents and identifying statistically over-represented words in a corpus when compared to a background set of texts. It enables developers to perform comparative text analysis and keyword discovery using both algorithmic and statistical methods.

## Plugin Architecture

All topwords-related classes inherit from a common base class, `TopwordsPlugin`, which provides a consistent API (such as `to_dict()`, `to_df()`, and `to_list()`) for serialization and output. This makes it easy to add new analysis methods as plugins.

## spaCy Doc Extensions

Custom attributes are registered on spaCy `Doc` objects to store analysis results:

- `topwords`: Stores the top distinguishing words for a document.
- `keywords`: Stores extracted keywords for a document.

These attributes are set automatically by the analysis classes if you provide spaCy `Doc` objects.

## Classes

### `TextacyKeywords`

- Extracts representative keywords from a single text document using algorithms from the textacy library.
- Backend: textacy.extract.keyterms.
- Methods: Supports standard keyword extraction algorithms via the method parameter, including `textrank` and `sgrank`.
- Customization: The `topn` parameter allows you to specify the number of top keywords to return.
- Output: After calling the instance, use `.to_dict()`, `.to_df()`, or `.to_list()` to get the results in your preferred format.

__Parameters:__

- `text`: The raw text to analyze.
- `method`: `"textrank"` or `"sgrank"` (default: `"textrank"`).
- `topn`: Number of top keywords to return (default: 10).
- `tokenizer`: (optional) A `Tokenizer` instance.

__Usage:__

```python
from lexos.topwords import TextacyKeywords

kw = TextacyKeywords(text="Your text here", method="textrank", topn=5)
kw()  # Run the analysis
print(kw.to_dict())  # List of dicts
print(kw.to_df())    # DataFrame
print(kw.to_list())  # List of tuples
# The keywords are also set on the spaCy Doc as kw.doc._.keywords
```

### `ZTestTopwords`

- Identifies the most distinguishing words in a set of target documents compared to a background corpus using a Z-test for statistical significance.
- Backend: numpy for statistical calculations and spaCy (via the Lexos Tokenizer) for text processing.
- Method: A Z-test is used to compare the proportions of word frequencies between the target and background corpora.
- Preprocessing: Includes boolean options for `case_sensitive` analysis and for removing `stopwords`, `punct` (punctuation), and `digits`.
- Customization: The `topn` parameter controls how many of the most significant words are returned.
- Output: After calling the instance, use `.to_dict()`, `.to_df()`, or `.to_list()` to get the results in your preferred format.

__Parameters:__

- `target_texts`: List of target documents (strings).
- `background_texts`: List of background documents (strings).
- `topn`: Number of top words to return (default: 10).
- `case_sensitive`, `remove_stopwords`, `remove_punct`, `remove_digits`: Preprocessing options.
- `tokenizer`: (optional) A `Tokenizer` instance.
- `docs`: (optional) List of spaCy `Doc` objects to set results on.

__Usage:__

```python
from lexos.topwords import ZTestTopwords
import spacy

nlp = spacy.blank("en")
target_texts = ["This is a test.", "Another test document."]
background_texts = ["Background text here.", "More background docs."]
docs = [nlp(text) for text in target_texts]

ztest = ZTestTopwords(
    target_texts=target_texts,
    background_texts=background_texts,
    docs=docs
)
ztest()  # Run the analysis
print(ztest.to_dict())  # List of dicts
print(ztest.to_df())    # DataFrame
print(ztest.to_list())  # List of tuples
# Each doc in docs now has doc._.topwords set to the top words
```

## Extending

To add a new analysis method, inherit from `TopwordsPlugin` and implement your logic. You can also register new custom spaCy Doc extensions as needed.
