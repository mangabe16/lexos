# Scrubbing Texts

## The Scrubber Module

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

1. [Normalize](scrubber/normalize.md) components are used to manipulate text into a standardized form.
2. [Remove](scrubber/remove.md) components are used to remove strings and patterns from text.
3. [Replace](scrubber/replace.md) components are used to replace strings and patterns in text.
4. [Tags](scrubber/tags.md) components are used to remove and replace tags, elements, attributes, and their values in texts marked up in HTML or XML.

Follow these links to read about the functions in each of Scrubber's components.

!!! note "Developer's Note"
    Many of the functions and resources in Scrubber are built on top of the preprocessing functions in the Python <a href="https://github.com/chartbeat-labs/textacy/" target="_blank">Textacy</a> library, although sometimes with modifications. Textacy is installed with Lexos, so it can also be called directly where that is useful.

## Loading Scrubber Components

Components must be loaded before they can be used. We can load them individually, as in the first example below, or we can specify multiple components in a tuple, as in the second example. In both cases, the returned variable is a function, which we can then feed to a scrubbing pipeline.

There are several ways to load Scrubber components into a Python file. The simplest is to import the component directly:

```python
from scrubber.normalize import lower_case
```

It is also possible to load the entire registry, called `scrubber_components`, and get individual components as needed:

```python
# Load the Scrubber components registry
from lexos.scrubber import scrubber_components

# Load a single component from the registry
lower_case = scrubber_components.get("lower_case")
```

In addition, Lexos provides a `get_components()` helper function to load components from the registry:

```python
# Load the helper functions
from lexos.scrubber import get_components

# Load multiple components using the helper function
punctuation, remove_digits = get_components("punctuation", "digits")
```

!!! note
    The `get_components()` function will also accept lists and tuples of component names, as well as single component names. However, if you wish to get a single component, you can also import `get_component()` and use that function instead.

    Which method you choose will depend on your preference and your workflow. Getting components from the registry by their string names can be especially helpful when developing applications.

### Custom Scrubbing Components

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
    To use a custom scrubbing function, you must register it _before_ you call `get_component()` or `get_components()`.

## Using Components

Loaded component functions can be called like any normal function. For example:
`scrubbed_text = remove_digits("Lexos123", only=["2", "3"])` will return "Lexos1".

If you are intending to apply multiple components to a single piece of text, the more efficient method is to use a pipeline (discussed below).

## Scrubbing Pipelines

When scrubbing texts, we may need to apply Scrubber components in a particular order. For this, we need a scrubbing pipeline. This can be done with either the `make_pipeline()` function or the `Scrubber` class. Each of these methods is detailed below.

### Using `make_pipeline()`

To make a pipeline with the `make_pipeline()` function, we import the function and pass our compents to it in the order we want them to be implemented.

```python
from lexos.scrubber import make_pipeline
from lexos.scrubber import get_components

lower_case, punctuation = get_components("lower_case", "punctuation")

# Make the pipeline
pipe = make_pipeline(lower_case, punctuation)

# Scrub the text
scrubbed_text = pipe("Lexos is the number 12 text analysis tool!!")
```

This will return "lexos is the number 12 text analysis tool".

Many Scrubber functions take additional keyword arguments. To pass them to a pipeline, it is necesary to use the `functools.partial` function:

```python
from functools.import partial
from lexos.scrubber import get_components

lower_case, punctuation, digits = get_components("lower_case", "punctuation", "digits")

# Make the pipeline
pipe = make_pipeline(
    lower_case,
    punctuation,
    partial(remove_digits, only=["1"])
)

# Scrub the text
scrubbed_text = pipe("Lexos is the number 12 text analysis tool!!")
```

This will return "lexos is the number 12 text analysis tool".

You can also pass an iterable of pipes to `make_pipeline()` directly for single-use pipelines:

```python
pipes = (lower_case, punctuation)
pipe = make_pipeline(pipes)

scrubbed_text = pipe("Lexos is the number 12 text analysis tool!!")
```

You can also pass a pipeline to the `scrub()` function.

```python
from lexos.scrubber import scrub

pipes = (lower_case, punctuation)
pipeline = make_pipeline(pipes)

scrubbed_text = scrub(
    "Lexos is the number 12 text analysis tool!!",
    pipeline
)
```

### Using the `Scrubber` Class

The Lexos `Scrubber` class also provides a more object-oriented approach, which may be more useful for some workflows. Start by creating an instance of the class. You can then add components to the pipeline with the `add_pipe()` method:

```python
from lexos.scrubber import Scrubber

scrubber = Scrubber()
scrubber.add_pipe("lower_case")
```

Notice that if the input is the string name of a component, it will automatically be fetched from the registry.

The `add_pipe()` method can also take an iterable of components such as `["lower_case", "remove_digits"]`. If the function takes keyword arguments, it can be passed as a tuple with the keyword arguments in a dictionary:

```python
scrubber.add_pipe(["lower_case", ("remove_digits", {"only": ["1"]})])
```

It is also possible to pass a `functools.partial` object for somewhat cleaner syntax:

```python
from functools import partial
scrubber.add_pipe(["lower_case", partial(remove_digits, only=["1"])])
```

The components will be added to the pipeline in the order they are passed. You can insert components in particular positions at any time using the `first`, `last`, `before`, and `after` arguments:

```python
scrubber.add_pipe("remove_digits", first=True)
scrubber.add_pipe("remove_digits", last=True)
scrubber.add_pipe("remove_digits", before="lower_case")
scrubber.add_pipe("remove_digits", after="lower_case")
```

The `before` and `after` arguments can also take an integer indicating the position within the pipeline.

Once the pipeline is set up, you can scrub text with the `scrub()` method:

```python
scrubbed_text = scrubber.scrub("Lexos is the number 12 text analysis tool!!")
```

This returns "lexos is the number 2 text analysis tool!!".

It is also possible to apply the scrubbing pipeline to an iterable of texts using the `pipe()` method:

```python
texts = [
    "Lexos is the number 12 text analysis tool!!",
    "Lexos is the number 1 text analysis tool!!"
]
scrubbed_texts = scrubber.pipe(texts)
for text in scrubbed_texts:
    print(text)
```

!!! important
    The `pipe()` method returns a generator, so use `list(scrubber.pipe(texts))` if you need a list of texts.

Under the hood, `Scrubber` uses the `Pipe` class to manage the pipeline. Each component added to the pipeline is converted to a `Pipe` object, which has a string `name` attribute and an `opts` dictionary to store keyword arguments accepted by the component function. It also has a `__call__()` method that applies the component to the text. You can create a `Pipe` object directly and use it to scrub text:

```python
from lexos.scrubber import Pipe

text = "Number 12 is the best number!"
my_pipe = Pipe("digits", {"only": ["1"]})
my_pipe(text) # Returns "Number 2 is the best number!"
```

You can even create and apply your own pipeline:

```python
text = "Number 12 is the best number!"
pipes = [Pipe("lower_case"), Pipe("digits", {"only": ["1"]})]
for pipe in pipes:
    text = pipe(text) # Returns "number 2 is the best number!"
```

The Scrubber `add_pipe()` and `pipe()` methods also accept `Pipe` objects or iterables of `Pipe` objects, which in many use cases can be a more convenient way to manage a pipeline.

!!! note "Developer's Note"
    When a `Pipe` object is instantiated, it automatically validates that the `name` and `opts` are of the correct datatypes, that the registry has been imported, and that the specified component is in the registry. If is not, the `Pipe` object raises an error.

#### Managing the Pipeline

After the pipeline is set up, you can use the following methods to manage it:

1. `add_pipe()`: Add additional components to the pipeline.
2. `remove_pipe()`: Remove a pipe from the pipeline (takes the string name of the component or a list of component names).
3. `reset()`: Reset the pipeline to an empty list.

If an existing component is added to the pipeline, any options will be merged with the existing options.

The `pipe()` method allows the existing configuration to be overridden using the `disable` and `component_cfg` arguments. The `disable` argument takes a list of component names to disable, while the `component_cfg` argument takes a dictionary of component names and the options to override. Scrubbing will be applied to the text according to these settings, but the stored pipeline will not be modified. The code below provides some examples:

```python
scrubber = Scrubber()
scrubber.add_pipe([
    "lower_case",
    "punctuation",
    partial(remove_digits, only=["1"])
])

# Scrub the text using `pipe()` with the `digits` component disabled
scrubbed_text = list(scrubber.pipe([text], disable=["digits"]))[0]

# Returns "this is a sample text with some digits 12345 and some punctuation lets see how it works"

# Remove the "punctuation" component from the pipeline
scrubber.remove_pipe("punctuation")

# Scrub the text with `scrub()`
scrubbed_text = scrubber.scrub(text)

# Returns "this is a sample text with some digits: 2345 and some punctuation! let's see how it works.
```

Notice that the `disable` keyword in `pipe()` only applies to that operation. The `remove_pipe()` method removes a component from the pipeline entirely.
