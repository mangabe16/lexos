# README for Key Words in Context (kwic) Module


## Overview

This module is designed to find key words and their sorrunding contexts in a tokenized text input. This module is a wrapper of the implementation of kwic from the textacy module. 

## Features

The tokenizer module includes 1 class with 4 functions:
- `find()`
    - Find all instances of a keyword within a spaCy doc and the context of each instance of that keyword
    - Context is determined by number of characters
- `find_multiple_keywords()`
    - Find all instances of multiple keywrods within a spaCy doc and the context of each instance of each keyword
- `find_in_sentences()`
    - Find all instances of a keyword and return the sentences that they appear in as context
    - Context window size is always the entire sentence length - cannot be modified
- `find_tokens()`
    - Find all instances of a keyword within a spaCy doc and the context of each instance of that keyword
    - Context is determined by number of tokens

All functions contain at least 4 parameters:
- `doc`: The spaCy doc(s) to search for the keyword in. Can be a single doc, or a list of docs.
- `keyword`: The keyword to search for. Most functions expect a string, but `find_multiple_keywords()` expects a list of strings.
- `ignore_case`: Weather to ignore the case of the word in the search.
- `dataframe_format`: Weather the output should be in the format of a pandas dataframe. If set to false, will return a list of tuples instead.

## Dependencies

The kwic module is a part of the greater Lexos library -- please see the Lexos README for all dependencies.

The kwic module is a wrapper class of the kwic class from the textacy module. The texacty kwic method, `keyword_in_context`, requires a spaCy doc or a regrex expression as input. While these libraries are included as a part of ther Lexos API, please ensure these libraries are installed to enable full functionality.

The kwic module only takes in spaCy docs as input. To convert a string or other input to a spaCy doc, users can use the `tokenizer` module of the Lexos library - please see that module's README and tutorial for usage.

## In-Depth Tutorial

For a guide on how to use the kwic module and more details about its utility, please refer to the Jupyter notenook **`tutorial.ipynb`**, which includes executable code examples on how to use the module. 

## Running Tests (For Developers)

In order to run a pytest test suite to ensure 100% coverage, please use the following line in the terminal: 

```bash
# Run the kwic module test
uv run pytest tests/kwic/kwic_test.py
```