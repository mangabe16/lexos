# README for DTM module

## Overview
This module provides tools for working with Document-Term Matrices (DTM) in Python. It is part of the Lexos project and is designed to facilitate text analysis workflows.

## Features
- Creation and manipulation of DTMs
- Support for various input formats
- Integration with other Lexos modules
- Calculate term percentages with configurable rounding
- Conversion to Pandas dataframe

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

## Running Tests
This repository also contains a comprehensive test suite which was found to have 100% coverage. To execute the pytest suite to validate functionality:

```bash
# Run all tests in the DTM module
uv run pytest tests/dtm/test_dtm.py
```

