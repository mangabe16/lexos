# README for Scrubber Module

## Overview

The Scrubber module provides a flexible, pipeline-based system for text cleaning and normalization as part of the Lexos project. It enables users to preprocess text by applying a customizable sequence of "scrubber components" (pipes) to remove, replace, or normalize elements such as punctuation, digits, whitespace, and more.

## Features

- Modular pipeline for text scrubbing
- Built-in registry of reusable scrubber components
- Easy addition and removal of pipeline components
- Support for custom components and configuration
- Integration with other Lexos modules
- Batch processing of texts via generator interface
- Robust error handling

## Dependencies

The Scrubber module is part of Lexos. Please follow the main Lexos README to set up the virtual environment and install all necessary libraries.

- pydantic
- catalogue
- spaCy (for some advanced components)
- lexos

## Quick Start

Create a `Scrubber` object, add components to the pipeline, and scrub your text:

```python
from lexos.scrubber.scrubber import Scrubber

scrubber = Scrubber()
scrubber.add_pipe("lower_case")
scrubber.add_pipe("digits")  # Removes digits

text = "Lexos is the number 12 text analysis tool!!"
cleaned = scrubber.scrub(text)
print(cleaned)  # Output: "lexos is the number  text analysis tool!!"
```

You can also process multiple texts at once:

```python
texts = [
    "Lexos is the number 12 text analysis tool!!",
    "Lexos is the number 1 text analysis tool!!"
]
for cleaned in scrubber.pipe(texts):
    print(cleaned)
```

## Customizing the Pipeline

Add, remove, or reorder components using `add_pipe`, `remove_pipe`, and the `first`, `last`, `before`, and `after` arguments:

```python
scrubber.add_pipe("punctuation", last=True)
scrubber.remove_pipe("digits")
```

You can also pass options to components:

```python
scrubber.add_pipe(("digits", {"only": ["1"]}))  # Only remove the digit "1"
```

## Using the Registry Directly

Access individual scrubber functions from the registry:

```python
from lexos.scrubber.registry import scrubber_components
lower_case = scrubber_components.get("lower_case")
print(lower_case("Lexos"))  # Output: "lexos"
```

Register your own component:

```python
def title_case(text: str) -> str:
    return text.title()

scrubber_components.register("title_case", func=title_case)
```

## Batch Scrubbing with `pipe()`

The `pipe()` method returns a generator. To get all results as a list:

```python
results = list(scrubber.pipe(texts))
```

## One-off Scrubbing with `scrub()`

For single-use pipelines, use the `scrub` function:

```python
from lexos.scrubber.scrubber import scrub

pipeline = ["lower_case", ("digits", {"only": ["1"]}), "punctuation"]
result = scrub("Lexos is the number 12 text analysis tool!!", pipeline)
```

## Testing

To run tests for the Scrubber module:

```bash
uv run pytest tests/scrubber/
```

## Further Reading

- [Scrubber Documentation](https://scottkleinman.github.io/lexos/api/scrubber/)
- See `scrubber_tutorial.ipynb` in this directory for interactive examples.

---