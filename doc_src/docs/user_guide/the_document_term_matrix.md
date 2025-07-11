# The Document-Term Matrix

## Overview

!!! important
    This page is currently under construction.

## About the Document-Term Matrix

A document-term matrix (DTM) is the standard interface for analysis and information of document data. It consists in its raw form of a list of token counts per document in the corpus. Each unique token form is called a term. Thus it is really a list of term counts per document, arranged as matrix.

Producing a DTM is easy with Lexos. All you need a is list of document tokens and a list of labels for each document. In the example below, we will use spaCy docs as the input since we can interate through their tokens just like a list.

```python
from lexos.dtm import DTM
from lexos.tokenizer import Tokenizer

# Define some texts and their labels
texts = [
    "Our first text.",
    "Our second text.",
    "Out third text."
]
labels = ["Doc1", "Doc2", "Doc3"]

# Tokenize the texts
tokenizer = Tokenizer()
docs = list(tokenizer.make_docs(texts=texts))

# Create a Document-Term Matrix (DTM)
dtm = DTM()
dtm(docs=docs, labels=labels)
```

If we did not want to use spaCy docs, we would need to have a list containing lists of tokens like this:

```python
docs = [
    ["Our", "first", "text"],
    ["Our", "second", "text"],
    ["Our", "third", "text"]
]
```

!!! note
    Lexos uses Textacy's `[Vectorizer](https://textacy.readthedocs.io/en/latest/api_reference/representations.html#textacy.representations.vectorizers.Vectorizer)` is the default vectorizer. It is possible to use Textacy directly to produce a DTM. For instance, the following method will produce a a DTM containing the raw term counts for each document.

    ```python
    from textacy.representations.vectorizers import Vectorizer
    vectorizer = Vectorizer(tf_type="linear", idf_type=None, norm=None)
    tokenized_docs = []
    for doc in docs:
        tokenized_docs.append(token.text for token in doc)
    vectorizer.fit_transform(tokenized_docs)
    ```

    Using the Lexos `DTM` class allows you to swap in your own custom vectorizer and gives access to additional helper methods such as `to_df()` to output the DTM as a pandas DataFrame.

## Understanding the Vectorizer

When you create an instance of the `DTM` class, you automatically assign it a vectorizer. By default, this is Textacy's `Vectorizer` class. Here's how it works:

- The `Vectorizer` scans all documents to build a vocabulary of unique **terms** (token forms).
- It then counts the occurrences of each term in each document, resulting in a sparse matrix where rows represent documents and columns represent terms.

!!! note "About Sparse Matrixes"
    Since each document only contains a small subset of all terms in the corpus, a document-term matrix can be very large and mostly filled with zeros. A sparse matrix is highly efficient for storage and computation, especially with large corpora, because it only stores nonzero values. Lexos uses data structures from the `scipy.sparse` library to store the DTM as a sparse matrix to make computations faster, which allows you to work with large corpora without running into memory issues. When you convert the DTM to a pandas DataFrame for analysis or visualization, the underlying data remains efficient and scalable You can learn more about the <code><a href="https://docs.scipy.org/doc/scipy/reference/sparse.html" target="_blank">scipy.sparse</a></code> library.

    If you need a dense (regular) matrix for certain operations or compatibility with other libraries, you can convert the sparse DataFrame to a dense one by calling:

    ```python
    dense_df = dtm.to_df().sparse.to_dense()
    ```

    Be aware that this may use a lot of memory for large corpora.

### Culling the DTM

In many cases, you will want to cull terms from your DTM in order to reduce the size of the data or to remove terms which you think might not be meaningful for your research. A common form of culling is to restrict the data to the *n* most-frequently occurring terms. You can do this with the `max_n_terms` parameter. You can also restrict your data to terms occurring in a minimum number of documents with `min_df` or a maximum number of documents with `max_df`. Here is an example using all three:

```python
dtm = DTM(max_n_terms=100, min_df=2, max_df=5)
```

Depending on your workflow, you can also configure the vectorizer directly or when you call the `DTM` instance. Here are some examples show the three alternative ways to do it:

```python
# Configure the DTM instance
dtm = DTM(max_n_terms=100)

# Configure the DTM tokenizer directly
dtm.vectorizer.min_df=2

# Set the parameters when calling the DTM instance
dtm(docs=docs, labels=labels, max_df=5)
```

Feel free to use whichever approach you find most comfortable.

!!! note
    A further method of limiting the vocabulary is to provide a list of specific terms to be included in the matrix using the `vocabulary_terms` parameter.

### Normalizing the Values

The vectorizer is configured by default to generate a matrix of raw counts. However, it can often be beneficial to normalize the values in some way such as by calculating the term's frequency in proportion to all terms in the corpus. Or, if your documents vary in length, it can be beneficial to calculate the <a href="" target="_blank">term frequency-inverse document frequency (TF-IDF)</a>. The vectorizer allows you to do that with the `tf_type`, `idf_type`, `dl_type`, and `norm` parameters, all of which can be set using one the three methods discussed in the previous section.

!!! note: "TO DO"
    Add more discussion on the most common settings. Notes below:

- `tf_type` controls how term frequencies are calculated (e.g., raw counts, log-scaled, binary presence/absence). Options: `"linear"`, `"sqrt"`, `"log"`, `"binary"`.
- `idf_type` controls inverse document frequency type, how document frequency scaling is applied (for TF-IDF weighting). Options: `None`, `"linear"`, `"sqrt"`, `"log"`.
- `dl_type`: Controls normalization based on document length. Options: `None`, `"linear"`, `"sqrt"`, `"log"`.
- `norm`: Applies vector normalization. Options: `None`, `"l1"`, `"l2"`. Normalizes the resulting vectors (rows) to unit length. L1 normalization scales the term frequencies in each document so that the sum of the absolute values equals 1. This means each document vector is divided by the sum of its term frequencies, turning the values into proportions that sum to 1. L2 normalization scales the term frequencies so that the sum of the squares of the values equals 1 (i.e., the Euclidean norm is 1). This is useful for algorithms that are sensitive to the length of the document vectors, such as cosine similarity.

## Getting Term Counts and Frequencies

Once you have generated your DTM, you can extract useful information from its properties:

- `DTM.shape`: returns a tuple with the width and height of the matrix.
- `DTM.sorted_terms_list`: Returns a sorted list of terms in the DTM.
- `sorted_term_counts`: Returns a sorted dictionary of terms and their total counts across all documents in the DTM.

!!! note
    By default, terms are sorted according to the rules of language used by your operating system. You can set the DTM to use a different sorting algorithm with the `alg` keyword, which takes a `natsorted.ns.LOCALE` object. For further information, see the <a href="https://natsort.readthedocs.io/en/5.1.0/ns_class.html" target="_blank">natsort</a> documentation.

Perhaps the most useful method of the `DTM` class is `to_df()`, which converts the matrix to a pandas DataFrame for display or for further manipulation. As a DataFrame, the output can be modified using the full range of options available in the pandas API. However, `to_df()` provides parameters that can ease the process:

- `by`: The term or terms to sort by.
- `ascending`: Whether to sort by ascending values.
- `as_percent`: Whether to convert counts to percentages.
- `rounding`: The number of digits after the decimal point to include.
- `transpose`: Whether to pivot the rows and columns in the matrix.
- `sum`: Add a column showing the sum of each row.
- `mean`: Add a column showing the mean of each row.
- `median`: Add a column showing the median of each row.

The labels are human-readable names for the documents which would otherwise be referenced by numeric indices.

## Visualising the DTM

Once a document-term matrix table has been generated as a pandas dataframe, it becomes possible to use any of the [`pandas.DataFrame.plot`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html) methods, or to export the data for use with other tools. Here is an example using `matplotlib` and the `seaborn` library.

```python
import seaborn as sns
import matplotlib.pyplot as plt

top_n = 20
term_totals = df.sum(axis=0).sort_values(ascending=False)[:top_n]

if term_totals.empty:
    print("No term frequencies to plot. The DataFrame may be empty or contain no data.")
else:
    sns.barplot(
        x=term_totals.index,
        y=term_totals.values,
        hue=term_totals.index,
        palette="viridis",
        legend=False
    )
    plt.title(f"Top {top_n} Most Frequent Terms", fontsize=18, weight="bold")
    plt.ylabel("Total Frequency", fontsize=14)
    plt.xlabel("Term", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()
```

However, the Lexos API has two built-in visualisations: word clouds and bubble charts. Word clouds can be generated for the entire DTM or for individual documents. Multiple word clouds arrange for comparison are referred to as multiclouds. For information on generating these and other visualizations, see the [Visualization page](../visualization/).

## Using `scikit-learn` Vectorizers

The machine-learning library `scikit-learn` (`sklearn`) provides methods for performing many common types of statistical analysis. When using this method, it may be preferable to generate the matrix as part of a pipeline that includes its <code><a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html" target="_blank">CountVectorizer</a></code> class (or similar vectorizers like <code><a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html" target="_blank">TfidfVectorizer</a></code>), rather than Textacy's `Vectorizer` class. The example below shows how this can be achieved whilst still leveraging language-specific knowledge available in a document tokenised with a language model.

```python
import spacy
from sklearn.feature_extraction.text import CountVectorizer

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Create a custom spaCy tokenizer function
def spacy_tokenizer(text):
    return [token.text for token in nlp(text)]

# Note that the tokenizer function can perform additional tasks:
def spacy_tokenizer(text):
    return [token.lemma_ for token in nlp(text) if token.pos_ == "NOUN"]

# Instantiate CountVectorizer with the custom tokenizer
vectorizer = CountVectorizer(tokenizer=spacy_tokenizer)

# Example text data
documents = ["This is the first document.", "This document is the second document.", "And this is the third one.", "Is this the first document?"]

# Fit and transform the documents
matrix = vectorizer.fit_transform(documents)

# Print the vocabulary
print(vectorizer.get_feature_names_out())

# Print the document-term matrix
print(matrix.toarray())
```

**Explanation:**

- The `spacy_tokenizer` function processes the input text using the loaded spaCy model. It then extracts the attribute of each token, returning a list of tokens.
- You then create a `CountVectorizer` object and tells it to use your custom function for tokenization instead of its default tokenizer to build the vocabulary and transform the documents into a matrix of token counts. Note that the output matrix will be a sparse representation of the token counts for efficiency.

Here is an example of how this procedure would be used for training a scikit-learn logistic regression model:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define the steps of the pipeline
pipeline_steps = [
    ('vectorizer', CountVectorizer()),
    ('logistic_regression', LogisticRegression())
]

# Create the pipeline
model_pipeline = Pipeline(pipeline_steps)

# Now, model_pipeline can be used like any other scikit-learn estimator
model_pipeline.fit(X_train, y_train)
predictions = model_pipeline.predict(X_test)
```
