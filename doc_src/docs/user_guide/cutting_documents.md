# Cutting Documents

## Overview

!!! important
    This page is currently a reproduction of the tutorial notebook. Complete documentation will be added soon.

In this tutorial, we demonstrate how to use two `cutter` classes for chunking text:

- `TextCutter` is used for raw strings or file inputs.
- `TokenCutter` is used for tokenized `spaCy` documents.

We’ll use Jane Austen’s *Pride and Prejudice* as our example dataset and try different methods of splitting it into smaller segments.

## Import Modules and Load Sample Text

Let’s import the necessary modules and read the first few lines of our sample text. We’re using a plain `.txt` file of *Pride and Prejudice*.

The first thing we’ll do is preview the first 500 characters to understand the structure of the text.

```python
import re
import os
import spacy

from lexos.cutter.text_cutter import TextCutter
from lexos.cutter.token_cutter import TokenCutter
from lexos.milestones.string_milestones import StringSpan
from pathlib import Path

# Get the directory of the current notebook
notebook_dir = Path().resolve()
# You can change this to the directory where your text file is located
# For example, if your text file is in a folder named 'data' in the current directory:
file_path = notebook_dir / "Austen_Pride.txt"
with open(file_path, "r", encoding="utf-8") as f:
    austen_text = f.read()

print(austen_text[:500])
```

## Cutting with `TextCutter`

`TextCutter` is used to split plain text strings or files into smaller chunks. It supports multiple cutting strategies, including byte-size chunks, fixed number of parts, line-based segments, and custom span-based splitting.

`TextCutter` has two main cutting methods:

- `split()` : Splits a string or file by byte size, number of chunks, or line count.
- `split_on_milestones()` : Splits a string or file using marked milestone spans.

We'll explore each of these in turn using sample text from *Pride and Prejudice*.

### `TextCutter.split()`

Splits text or file contents based on byte size, number of chunks, or line-based chunks.

**Arguments:**

- `chunksize` : Number of bytes per chunk (default = 1,000,000).
- `newline` : If `True`, split using number of lines instead of byte size.
- `file` : If `True`, treat each source as a file path.

#### Example: Split by Byte Size

This example splits a text into chunks of 100 bytes each.

```python
text_cutter_1 = TextCutter()
text_cutter_1.split(
    austen_text,
    chunksize=100,
    names=["Pride_Austen_ChunkSize"]
    )

for i, chunk in enumerate(text_cutter_1.chunks[0][:3]):
    print(f"Chunk {i+1}:\n{chunk}\n")
```

#### Example: Split by Number of Chunks

Here we split the text into 3 nearly equal parts by setting `n=3`.

```python
text_cutter_2 = TextCutter()
text_cutter_2.split(
    austen_text,
    n=3,
    names=["Pride_Austen_ByChunks"]
    )

for i, chunk in enumerate(text_cutter_2.chunks[0]):
    print(f"Chunk {i+1}:\n{chunk}\n")
```

#### Example: Split by Number of Lines

You can use `newline=True` to divide the text into line-based segments. This is useful for preserving sentence or paragraph boundaries.

```python
text_cutter_3 = TextCutter()
text_cutter_3.split(
    austen_text,
    n=2,
    newline=True,
    names=["Pride_Austen_Lines"]
    )

for i, chunk in enumerate(text_cutter_3.chunks[0]):
    print(f"Chunk {i+1}:\n{chunk}\n")
```

### `TextCutter.split_on_milestones()`

Splits a text or file at custom-defined spans called milestones. These are defined using `StringSpan` objects, which mark specific ranges of text (e.g., where a chapter starts).

**Arguments:**

- `milestones` : A list of `StringSpan` objects indicating split points.
- `keep_spans` : Determines whether to retain the span text:
  - `'preceding'` attaches it to the chunk before,
  - `'following'` attaches it to the chunk after,
  - `False` (default) removes it.
- `file` : Set to `True` if sources are files.

#### Example: Split on Milestones

This example uses the word "Chapter" as a recurring span. Each chunk begins with the "Chapter" label by setting `keep_spans='following'`.

```python
# Find "Chapter" occurrences
milestones = []
for i in range(len(austen_text)):
    if austen_text[i:i+7] == "Chapter":
        milestones.append(StringSpan(text="Chapter", start=i, end=i+6))

text_cutter_4 = TextCutter()
text_cutter_4.split_on_milestones(
    austen_text,
    milestones=milestones,
    names=["Pride_Austen_Milestones"],
    keep_spans='following',
    file=False
)

for i, chunk in enumerate(text_cutter_4.chunks[0][:3]):
    print(f"Chunk {i+1}:\n{chunk[:200]}\n")  # preview first 200 chars
```

### Saving Chunks with `TextCutter`

After splitting text using `TextCutter`, you can export the chunks to disk using the `.save()` method.

Each chunk will be saved as a `.txt` file using the format:

```bash
    <docname>_<chunk number>.txt
```

**Arguments:**

- `output_dir` : The folder to save the files in.
- `names` : Optional list of names for the documents.

```python
# Make the output directory if it doesn't exist
os.makedirs("text_chunks", exist_ok=True)
# Save line-based text chunks
text_cutter_3.save(output_dir="text_chunks", names=["Austen_Lines"])
```

### Merging Chunks with `TextCutter`

After splitting text, you can recombine the chunks into a single string using `.merge()`.

This method simply joins all the chunks using a space (or a custom separator if specified).

**Arguments:**

- `chunks` : A list of strings.

```python
# Merge text chunks back into one string
merged_text = text_cutter_3.merge(text_cutter_3.chunks[0])
print(merged_text[:500])  # Preview the first 500 characters
```

### Exporting Chunks with `TextCutter.to_dict()`

The `.to_dict()` method converts the chunks into a Python dictionary where:

- The **keys** are the names of the documents
- The **values** are lists of string chunks

This is useful for programmatic access or for converting the results to JSON or other data structures.

**Arguments:**

- `names` : Optional list of document names to use as keys.

```python
# Export line-based text chunks as a dictionary
text_dict = text_cutter_3.to_dict()

# Preview the dictionary keys and number of chunks
for name, chunks in text_dict.items():
    print(f"{name}: {len(chunks)} chunks")
```

## Cutting with `TokenCutter`

`TokenCutter` is used to split spaCy `Doc` objects into smaller segments. It provides multiple token-level cutting strategies including token count, sentence count, line breaks, and custom milestones.

`TokenCutter` has four main methods:

- `split()` : Splits spaCy docs by number of tokens.
- `split_on_sentences()` : Splits spaCy docs by number of sentences.
- `split_on_linebreaks()` : Splits spaCy docs by number of lines.
- `split_on_milestones()` : Splits spaCy docs using manually tagged tokens.

We'll explore each method using a spaCy-parsed version of *Pride and Prejudice*.

```python
# Load and prepare the spaCy Doc from the Austen text
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

doc = nlp(austen_text)  # This is the tokenized version of the Austen text
```

### `TokenCutter.split()`

Splits a spaCy Doc into segments of `chunk_size` tokens.

**Arguments:**

- `chunk_size` : Number of tokens per chunk (default = 1000).
- `overlap` : Number of tokens to repeat between adjacent chunks.

```python
token_cutter_1 = TokenCutter()
token_cutter_1.split(
    doc,
    chunk_size=25,
    names=["Austen_Tokens"]
    )

for i, chunk in enumerate(token_cutter_1.chunks[0][:3]):
    print(f"Chunk {i+1}:\n{chunk.text}\n")
```

### `TokenCutter.split_on_sentences()`

Splits a spaCy Doc into chunks of `n` sentences.

**Arguments:**

- `n` : Number of sentences per chunk.
- `overlap` : Number of tokens to repeat between chunks.

```python
token_cutter_2 = TokenCutter()
token_cutter_2.split_on_sentences(
    doc,
    n=1,
    names=["Austen_Sents"]
    )

for i, chunk in enumerate(token_cutter_2.chunks[0][:3]):
    print(f"Chunk {i+1}:\n{chunk.text}\n")
```

### `TokenCutter.split_on_linebreaks()`

Splits a spaCy Doc into chunks of `n` lines based on newline characters (`\\n`).

**Arguments:**

- `n` : Number of lines per chunk.

```python
token_cutter_3 = TokenCutter()
token_cutter_3.split_on_linebreaks(
    doc,
    n=2,
    names=["Austen_Lines"]
    )

for i, chunk in enumerate(token_cutter_3.chunks[0][:3]):
    print(f"Chunk {i+1}:\n{chunk.text}\n")
```

### `TokenCutter.split_on_milestones()`

Splits a spaCy Doc at custom token-level spans (e.g. `Span`s labeled “Chapter”).

**Arguments:**

- `milestones` : List of spaCy `Span`s (manually tagged tokens).
- `keep_spans` : Whether to keep milestone in chunk (`'preceding'`, `'following'`, or `False`)

```python
# Use regex to find all instances of "Chapter" (case-insensitive)
matches = [m.start() for m in re.finditer(r"\b[Cc]hapter\b", doc.text)]

# Create spans from match positions
milestones = [doc.char_span(start, start + 7) for start in matches]
milestones = [m for m in milestones if m is not None]  # Filter out failed spans

# Initialize TokenCutter and split on the detected spans
token_cutter_4 = TokenCutter()
token_cutter_4.split_on_milestones(
    doc,
    milestones,
    keep_spans='following',
    names=["Austen_Milestones"]
)

# Preview first few chunks
for i, chunk in enumerate(token_cutter_4.chunks[0][:3]):
    print(f"Chunk {i+1}:\n{chunk.text[:200]}\n")
```

### Saving Chunks with `TokenCutter`

For tokenized documents, use `.save_text()` to export chunks of spaCy `Doc` objects as plain `.txt` files.

This is useful when you want to inspect or reuse the text content outside of Python.

**Arguments:**

- `output_dir` : The folder to save the files in.
- `strip_chunks` : Whether to trim whitespace from chunk edges (default = True).

```python
# Make the output directory if it doesn't exist
os.makedirs("token_chunks", exist_ok=True)
# Save sentence-based token chunks
token_cutter_2.save_text(output_dir="token_chunks", names=["Austen_Sents"])
```

### Merging Chunks with `TokenCutter`

For `TokenCutter`, the `.merge()` method joins a list of spaCy `Doc` chunks back into a single `Doc` object.

This is useful if you want to reverse the chunking operation and analyze the full text again.

**Arguments:**

- `chunks` : A list of spaCy `Doc` objects.

```python
# Merge tokenized sentence chunks into one spaCy Doc
merged_doc = token_cutter_2.merge(token_cutter_2.chunks[0])
print(merged_doc.text[:500])  # Preview the first 500 characters
```

### Exporting Chunks with `TokenCutter.to_dict()`

Similar to `TextCutter`, the `.to_dict()` method for `TokenCutter` returns a dictionary where:

- Each **key** is a document name
- Each **value** is a list of `spaCy.Doc` objects

This allows for structured access to tokenized chunks in downstream workflows.

**Arguments:**

- `names` : Optional list of document names to use as keys.

```python
# Export sentence-based token chunks as a dictionary
token_dict = token_cutter_2.to_dict()

# Preview the dictionary keys and number of chunks
for name, chunks in token_dict.items():
    print(f"{name}: {len(chunks)} chunks")
```

