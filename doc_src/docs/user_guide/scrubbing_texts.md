# Scrubbing Texts

## Overview

Scrubber can be defined as a _destructive_ preprocessor. In other words, it changes the text as loaded in ways that potentially make mapping the results onto the original text impossible. It is therefore best used before other procedures so that the scrubbed text is essentially treated as the "original" text. This differs from the [Tokenizer](user_guide/tokenizing_texts.md), which divides the text into "tokens" (often words) without destroying the original text.

The Scrubber module has the following features:

- Modular pipeline for text scrubbing
- Built-in registry of reusable Scrubber component functions
- Easy addition and removal of pipeline components
- Support for custom components and configuration

Scrubbing works by applying a single function or a pipeline of functions to the text, with each function applied in the order given. Lexos has a registry of pre-built functions to perform many common pre-processing tasks. The use of the registry will be discussed further below.

!!! note
    In the Lexos web app, Scrubber is used to tokenize the text before any other scrubbing actions occur. In the Lexos Python package, these preprocessing and tokenization are kept strictly separate.

## Scrubber Components

Scrubber components are divided into three categories:

1. [Normalize](https://scottkleinman.github.io/lexos/api/scrubber/normalize/) components are used to manipulate text into a standardized form.
2. [Remove](https://scottkleinman.github.io/lexos/api/scrubber/remove/) components are used to remove strings and patterns from text.
3. [Replace](https://scottkleinman.github.io/lexos/api/scrubber/replace/) components are used to replace strings and patterns in text.
4. [Tags](https://scottkleinman.github.io/lexos/api/scrubber/tags/) components are used to remove and replace tags, elements, attributes, and their values in texts marked up in HTML or XML.

Follow these links to read about the functions in each of Scrubber's components.

## Loading Scrubber Components

Components must be loaded before they can be used. We can load them individually, as in the first example below, or we can specify multiple components in a tuple, as in the second example. In both cases, the returned variable is a function, which we can then feed to a scrubbing pipeline.

As a reminder, we need to load the scrubber components registry with

```python
# Load the Scrubber components registry
from lexos.scrubber.registry import scrubber_components

# Load a single component from the registry
lower_case = scrubber_components.get("lower_case")
```

Lexos also provides helper functions to load components from the registry, which can be used like this:

```python
# Load the helper functions
from lexos.scrubber.registry import load_component, load_components

# Load a single component using the helper function
lower_case = load_component("lower_case")

# Load multiple components using the helper function
punctuation, remove_digits = load_components(("punctuation", "digits"))
```

## Using Components

Loaded component functions can be called like any normal function. For example:
`scrubbed_text = remove_digits("Lexos123", only=["2", "3"])` will return "Lexos1".

If you are intending to apply multiple components to a single piece of text, the more efficient method is to use a pipeline.

## Making a Pipeline

Now let's make the pipeline. We simply feed our component function names into the `make_pipeline()` function in the order we want them to be implemented. Notice that `remove_digits` has to be passed through the `pipe()` function. This is because the `digits` function requires extra arguments, and `pipe()` allows those arguments to be passed to the main pipeline function.

```python
# Make the pipeline
scrub = make_pipeline(
    lower_case,
    punctuation,
    pipe(remove_digits, only=["1"])
)
```

The value returned is a function that implements the full pipeline when called on a text, as shown below.

```python
# Scrub the text
scrubbed_text = scrub("Lexos is the number 12 text analysis tool!!")
```

This will return "lexos is the number 2 text analysis tool".

You can also use the `scrub()` function directly for single-use pipelines:

```python
pipeline = ["lower_case", ("digits", {"only": ["1"]}), "punctuation"]
result = scrub("Lexos is the number 12 text analysis tool!!", pipeline)
```

## Custom Scrubbing Components

Users can write and use custom scrubbing functions. The function is written like a normal function, and to use it like a scrubber component it must be added to the registry. Below is an example with a custom `title_case` function.

```python
# Define the custom function
def title_case(text: str) -> str:
    """Our custom function to convert text to title case."""
    return text.title()

# Register the custom function
scrubber_components.register("title_case", func=title_case)
```

!!! important
    To use a custom scrubbing function, you must register it _before_ you call `load_component()` or `load_components()`.
