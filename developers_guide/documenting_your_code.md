# Documenting Your Code

Clear code documentation is essential for maintaining and understanding the codebase. Docstrings are a key part of this documentation, as they provide a way to describe the purpose and behavior of functions, classes, and modules. They are also used to generate API documentation, which is important for users who want to understand how to use the Lexos library.

Ruff has a large number of built-in rules which will be enforced when you perform linting. For instance, Python modules should be documented with a docstring at the top of the file that contains the name of the file and ends with a period.

This file contains a few additional useful notes and links to help you think about your docstrings.

## Docstring Style

The Lexos project follows the [Google Style Python Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings. This is a widely used style guide that provides a consistent format for writing docstrings in Python code. It is recommended to follow this style guide for all docstrings in the Lexos project.

The basic structure of a docstring in the Google Style is as follows:

```python
def function_name(param1: int, param2: str) -> bool:
    """Summary of the function.

    Args:
        param1 (int): Description of parameter 1.
        param2 (str): Description of parameter 2.

    Returns:
        bool: Description of return value.

    Raises:
        ValueError: Description of the error condition.
    """
    # Function implementation goes here
```

Only the "Args" section is required. Note that type hints should be reproduced in the docstring. The "Returns" section should be provided if the function returns a value, and the "Raises" section should only be included if the function raises exceptions. The "Summary" line should be a short, one-line description of the function's purpose. A "Description" section can provide additional details about the function's behavior, if necessary.

For test functions, only a "Summary" line is required. The other sections can be provided optionally if they help to explain the function's behaviour.

## Documenting Pydantic Models Using the `Field` Function

Pydantic models are used to define data structures in Python. When using Pydantic, you can use the `Field` function to provide additional metadata for model fields. This metadata can include descriptions, default values, and validation constraints. Here is an example of how to use the `Field` function to document a Pydantic model:

```python
from pydantic import BaseModel, Field
from typing import Optional

class Person(BaseModel):
    name: str = Field(default="John Doe", description="The name of the person.")
    age: Optional[int] = Field(default=20, description="The age of the person.")
```

Note that, due to a misreading of the Pydantic documentation, many classes provide the description using the `json_schema_extra` keyword:

```python
class Person(BaseModel):
    name: str = Field(
        default="John Doe",
        json_schema_extra={"description": "The name of the person."}
    )
```

After our initial code review, we will be updating the code to make consistent use `description` keyword instead of `json_schema_extra` across the code base.

!!! Important
    Since the Pydantic `Field` function provides the same information as the docstring, the docstring should only contain a summary of the class and any additional information that is not already provided by the `Field` function.

## Docstrings and the Lexos API Documentation

The Lexos project uses [mkdocs](https://www.mkdocs.org/) to generate API documentation from directly docstrings. The API documentation is automatically generated from the docstrings in the codebase, so it is important to keep the docstrings up to date and consistent with the code.

## Computed Fields and `model_dump()`

Pydantic models can define computed or derived properties that are evaluated on demand. In Lexos, we intentionally use several computed behaviors (for instance, on `Record` objects) to provide convenience accessors such as `terms`, `tokens`, `num_terms` and `num_tokens`.

However, developers should be aware of two important caveats when using `model_dump()`:

- `model_dump()` may evaluate computed fields. If a computed property depends on runtime state (for example, `Record.terms` depends on the `content` being a parsed `Doc`), evaluating it via `model_dump()` may trigger exceptions such as `LexosException("Record is not parsed.")`.
- Computed fields may be expensive. If a computed field does heavy computation (e.g., building a DTM, calculating statistics, or serializing spaCy tokens), calling `model_dump()` on the object could result in unexpected slowdowns.

### Guidance and recommended patterns
-------------------------------

1. Guard model_dump() calls when the model may not be in a state that supports computed properties. If a model has a boolean state property such as `is_parsed` or `is_ready`, prefer checking it before calling `model_dump()` on that model. Example:

```python
if record.is_parsed:
    data = record.model_dump()
else:
    # build a safe representation instead of model_dump()
    data = {
        "id": str(record.id),
        "name": record.name,
        "is_active": record.is_active,
        "text": "",
        "terms": [],
        "tokens": [],
    }
```

1. Use `exclude` and `mode="json"` to explicitly prevent evaluation of computed fields where appropriate:

```python
meta = record.model_dump(exclude=["terms", "tokens", "text"], mode="json")
```

1. If you control a public API or export, prefer returning explicitly constructed dictionaries instead of blindly returning `model_dump()` output for objects with computed fields. This gives you fine-grained control over what is safe to serialize.

1. Add unit tests that assert `model_dump()` on objects does not implicitly evaluate computed fields that can raise; or, test the guarded pattern. For example, a test can verify that `to_df()` returns defaults instead of raising when records are unparsed.

1. Document computed-field behavior in the model's module. When defining a computed field in a Pydantic model, add a short explanation and, if it can raise, mention the conditions under which it will raise.

### Example: `Record` computed fields
-------------------------------
The `Record` model provides several computed properties that require the `content` to be a parsed `spacy.Doc`. Accessing properties such as `terms`, `tokens`, `num_terms`, and `num_tokens` will raise `LexosException("Record is not parsed.")` if `content` is not parsed. These fields should therefore be excluded in calls to `model_dump()` unless the caller is sure the record is parsed, or the call is guarded by `is_parsed` checks.

For an example of the guarded pattern in practice, see `Corpus.to_df` in `src/lexos/corpus/corpus.py`, which explicitly checks `record.is_parsed` before calling `record.model_dump()` and assembles a safe row for unparsed records instead.

### Why this matters for future contributions
-------------------------------
Adding new computed properties or changing the behavior of existing ones can introduce subtle bugs when other modules call `model_dump()` without guarding computed-property evaluation. By following the guidance above and documenting computed behaviors loudly (in code comments, docstrings, and in these developer guidelines), the risk of inadvertently triggering a computed-field evaluation that raises or is expensive is minimized.

If you are in doubt, ask in a code review or add a comment next to the `model_dump()` call explaining why it is safe (or why the caller must ensure certain state before calling it).
