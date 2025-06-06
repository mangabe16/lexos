# README for DTM module

## Overview

This module provides tools for working with Document-Term Matrices (DTM) in Python. It is part of the Lexos project and is designed to facilitate text analysis workflows.

## Features

- Creation and manipulation of DTMs
- Support for various input formats
- Integration with other Lexos modules
- Calculate term percentages with configurable rounding
- Conversion to Pandas dataframe
- Robust error handling

## Dependencies

The DTM module is part of lexos, follow the lexos README to install the virtual environment and all necessary libraries.

- numpy
- pandas
- scipy
- natsort
- pydantic
- spaCy
- textacy
- lexos

## In-Depth Tutorial

For a comprehensive, interactive guide on how to use the DTM module, including detailed explanations and runnable code examples, please refer to our Jupyter Notebook tutorial, also loacted in this directorty:

- **`tutorial.ipynb`**

This tutorial covers document preparation, advanced `Vectorizer` configurations, DTM rebuilding workflows, detailed `to_df()` usage, and how to verify code coverage.

## Running Tests

This repository also contains a comprehensive test suite which was found to have 100% coverage. To execute the pytest suite to validate functionality:

```bash
# Run all tests in the DTM module
uv run pytest tests/dtm/test_dtm.py
```
