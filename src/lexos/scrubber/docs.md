# Scrubbing Texts

[## About Scrubber](#about-scrubber)

Scrubber can be defined as a _destructive_ preprocessor. In other words, it changes the text as loaded in ways that potentially make mapping the results onto the original text potentially impossible. It is therefore best used before other procedures so that the scrubbed text is essentially treated as the "original" text. Unlike the web app, in the Lexos API `Scrubber` does not play a role in tokenisation such as by separating tokens by whitespace.

Scrubbing works by applying a single function or a pipeline of functions called "components" or "pipes" to the text. Lexos comes with a number of pre-built component functions, which can be accessed through its registry. To access these components, you need to import `scrubber_components`:

```python
from lexos.scrubber.registry import scrubber_components
```

Under the hood, the registry uses the Python [`catalogue`](https://github.com/explosion/catalogue) package to manage compnents. You can access individual functions with its `get()` method:

```python
lower_case = scrubber_components.get("lower_case")
lower_case("Lexos")  # Returns "lexos"
```

The `lexos.scrubber.registry` module also provides a `load_component()` function that does the same thing, as well as a `load_components()` function that loads multiple components passed as a tuple:

```python
lower_case = get_component("lower_case")
punctuation, remove_digits = get_components(("punctuation", "digits"))
```

You can add custom components to the registry with the `register()` method:

```python
def title_case(text: str) -> str:
    """Our custom function to convert text to title case."""
    return text.title() # Register the custom function
scrubber_components.register("title_case", func=title_case)
```

You can also do this with a decorator:

```python
@scrubber_components.register("title_case")
def title_case(text: str) -> str:
    """Our custom function to convert text to title case."""
    return text.title()
```

Scrubber expects all components to be in the registry. Each component must take a single string argument and return a single string. Components may also take additional arguments, which can be passed to the component function as keywords:

```python
@scrubber_components.register("remove_digits")
def remove_digits(text: str, only: list[str] = None) -> str:
    """Remove all digits from the text."""
    if only:
        return "".join([char for char in text if char not in only])
    return "".join([char for char in text if not char.isdigit()])
```

[## Scrubber Components](#scrubber-components)

Scrubber components are divided into three categories:

1. [Normalize](https://scottkleinman.github.io/lexos/api/scrubber/normalize/) components are used to manipulate text into a standardized form.
2. [Remove](https://scottkleinman.github.io/lexos/api/scrubber/remove/) components are used to remove strings and patterns from text.
3. [Replace](https://scottkleinman.github.io/lexos/api/scrubber/replace/) components are used to replace strings and patterns in text.

Follow these links to view all of the default scrubber components.

Component functions may be imported directly from the `lexos.scrubber` module and used on an ad hoc basis. For example, to import the `lower_case` component, you would use:

```python
from lexos.scrubber.normalize import lower_case
lower_case("Lexos")  # Returns "lexos"
```

However, in most cases, you will want to use the `Scrubber` class to manage your scrubbing pipeline.

[## The Scrubber Class](#the-scrubber-class)

The `Scrubber` class provides an object-oriented approach to managing a scrubbing pipeline. Start by creating an instance of the class.

```python
from lexos.scrubber import Scrubber
scrubber = Scrubber()
```

You can then add components to the pipeline with the `add_pipe()` method:

```python
scrubber.add_pipe("lower_case")
```

The `add_pipe()` method can also take an iterable of components such as `["lower_case", "remove_digits"]`. If the function takes keyword arguments, it can be passed as a tuple with the keyword arguments in a dictionary:

```python
scrubber.add_pipe(["lower_case", ("remove_digits", {"only": ["1"]})]
```

It is also possible to pass a `functools.partial` object:

```python
from functools import partial
scrubber.add_pipe(["lower_case", partial(remove_digits, only=["1"])]
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
scrubber.scrub("Lexos is the number 12 text analysis tool!!")  # Returns "lexos is the number 2 text analysis tool"
```

It is also possibly to apply the scrubbing pipeline to an iterable of texts using the `pipe()` method:

```python
texts = ["Lexos is the number 12 text analysis tool!!", "Lexos is the number 1 text analysis tool!!"]
scrubber.pipe(texts)  # Returns ["lexos is the number 2 text analysis tool", "lexos is the number 2 text analysis tool"]
```

Note that the `pipe()` method returns a generator, so to see the result, use `list(scrubber.pipe(texts))`.

[## The Pipe Class](#the-pipe-class)

The `pipe()` method has various options which will be discussed further below.

## [The `Pipe` Class](#the-pipe-class)

Under the hood, the `Scrubber` class uses the `Pipe` class to manage the pipeline. Each component added to the pipeline is converted into an instance of the `Pipe` class, which has a string `name` attribute and a dictionary `opts` attribute to store keyword arguments accepted by the component function. It also has a `__call__()` method that applies the component to the text. You can create a `Pipe` object directly and use it to scrub text:

```python
from lexos.scrubber import Pipe
text = "Number 12 is the best number!"
text = Pipe("remove_digits", {"only": ["1"]}) # Returns "Number 2 is the best number!"
```

You can even create and apply your own pipeline:

```python
from lexos.scrubber import Pipe
text = "Number 12 is the best number!"
pipes = [Pipe("lower_case"), Pipe("remove_digits", {"only": ["1"]})]
for pipe in pipes:
    text = pipe(text) # "number 2 is the best number!"
```

However, the `Scrubber` `add_pipe()` and `pipe()` methods also accept `Pipe` objects or iterables of `Pipe` objects, which in many use cases can be a more convenient way to manage a pipeline.

When a `Pipe` object is instantiated, it automatically validates that the `name` and `opts` are of the correct datatypes, that the registry has been imported, and that the specified component is in the registry. If they are not, it raises an error.

## [`Scrubber` Class Methods](#scrubber-class-methods)

After the pipeline is set up, you can use the following methods to manage it:

1. `add_pipe()`: Add additional components to the pipeline.
2. `remove_pipe()`: Remove a pipe from the pipeline (takes the string name of the component or a list of component names).
3. `reset()`: Resets the pipeline to an empty list.

If an existing component is added to the pipeline, any options will be merged with the existing options.

The `pipe()` method allows the existing configuration to be overridden using the `disable` and `component_cfg` arguments. The `disable` argument takes a list of component names to disable, while the `component_cfg` argument takes a dictionary of component names and the options to override. Scrubbing will be applied to the text according to these settings, but the stored pipeline will not be modified.

## [The `scrub` Function](#the-scrub-function)

The `scrub` function is a convenience function that allows you to scrub text without creating a `Scrubber` object.  It is particularly useful for one-off scrubbing tasks or for testing components. It takes a text and a pipeline (iterable of components) to apply to the text and returns the scrubbed text. The components can be be a variety of different types

- A function or a `functools.partial` object.
- A string name of a component in the registry.
- A tuple containing a string name of a component in the registry and a dictionary of keyword arguments.
- A `Pipe` object.

The code below demonstrates how to use the `scrub` function:

```python
from functools import partial
from lexos.scrubber.normalize import lower_case
from lexos.scrubber.registry import scrubber_components
from lexos.scrubber import Pipe, scrub

pipeline = [
    lower_case,
    partial(remove_digits, only=["1"]),
    Pipe("remove_digits", {"only": ["2"]}),
    ("remove_digits", {"only": ["3"]}),
    "remove_digits"],
    "upper_case"
]
text = "Banana1234"
scrub(text, pipeline)  # Returns "BANANA"
```

First the `lower_case` function is applied. Then the partial function removes only the number "1". Next, the `Pipe` object removes the number "2". The tuple removes the number "3". Then the "remove_digits" function is fetched from the registry and applied to remove all remaining digits. Finally, the "upper_case" function is fetched from the registry and used to convert the remaining string to uppercase.

Components in the pipeline are applied in order, and the order can be changed by manipulating the pipeline iterable.
