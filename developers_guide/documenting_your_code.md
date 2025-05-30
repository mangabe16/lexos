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
