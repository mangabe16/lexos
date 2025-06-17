# README for Key Words in Context (kwic) Module


## Overview

This module is designed to find key words and their sorrunding contexts in a tokenized text input. This module is a wrapper of the implementation of kwic from the textacy module. 

## Features

The tokenizer module includes 1 class with one function. The Kwic class has one function, `find()`, which finds the context of a desired key word. The amount of context in behind and ahead the key words can be chosen by the user, in terms of amount of characters. The `find()` function works with spaCy docs or regex expressions.

## Dependencies

The kwic module is a part of the greater Lexos library -- please see the Lexos README for all dependencies.

The kwic module is a wrapper class of the kwic class from the textacy module. The texacty kwic method, `keyword_in_context`, requires a spaCy doc or a regrex expression as input. While these libraries are included as a part of ther Lexos API, please ensure these libraries are installed to enable full functionality.


## Running Tests (For Developers)

In order to run a pytest test suite to ensure 100% coverage, please use the following line in the terminal: 

```bash
# Run the kwic module test
uv run pytest tests/kwic/kwic_test.py
```