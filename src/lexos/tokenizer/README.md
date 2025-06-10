# README for Tokenizer Module


## Overview

This module is designed to provide tools for tokenizing texts using natural language processing (NLP) models. 

## Features

The tokenizer module includes 2 files and 4 classes. 

- lexos.tokenizer
    - Tokenizer
        - Tokenizes texts using spaCy NLPs
        - Takes in raw text as input
        - Returns spaCy docs that contains the tokens of the given input text
        - Includes filtering of digits, punctuation, stopwords, etc.
        - Supports all spaCy NLP models
    - SliceTokenizer
        - Simple slice tokenizer
        - Can be used to generate character tokens/ngrams
    - WhitespaceTokenizer
        - Tokenizes on whitespace
- lexos.tokenizer.ngrams
    - Ngrams
        - Returns ngrams from a text, spaCy doc, or list of tokens
        - Includes filtering of digits, punctuation, stopwords, whitespace, etc.
        - User selected size of ngrms


## Dependencies

The tokenizer module is a part of the greater Lexos library -- please see the Lexos README for all dependencies.

The tokenizer module uses spaCy NLP models to tokenize input texts. These module must be installed locally in order for the tokenizer module to run correctly. By default, the `xx_sent_ud_sm` model is used. If another model is desired, those models must also be installed.

## In-Depth Tutorial

For a guide on how to use the tokenizer module and more details about its utility, please refer to the Jupyter notenook **`tutorial.ipynb`**, which includes executable code examples on how to use the module. 

## Running Tests (For Developers)

In order to run a pytest test suite to ensure 100% coverage, please use the following line in the terminal: 

```bash
# Run all tests in the tokenizer module
uv run pytest tests/tokenizer
```