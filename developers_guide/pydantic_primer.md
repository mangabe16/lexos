# A Short Primer on Pydantic

In Python, you define a class like this:

```python
class MyPythonClass:
    def __init__(self, value: int):
        self.value = value
```

In Pydantic, the class would like this:

```python
from pydantic import BaseModel

class MyPydanticClass(BaseModel):
    value: int
```

So far, so good. Pydantic looks like a Python dataclass and has much cleaner code. However, compare the following instantiations of our two classes:

```python
python_instance = MyPythonClass("1")
pydantic_instance = MyPydanticClass(value="1")
pydantic_instance = MyPydanticClass(value={"myvalue": 1})
```

The `python_instance` will not raise an error because there is no type checking. The first `pydantic_instance` will also raise an error because, by default, Pydantic attempts to coerce data into the expected data type (you can change this behaviour). By default, it knows to convert strings to integers. However, it will raise a `ValidaError` for the second `pydantic instance` since it doesn't know how to coerce dicts.

Note also that Pydantic requires keyword arguments when instantiating a class, and you cannot use positional arguments. This is a [design choice](https://github.com/pydantic/pydantic/issues/116) made by Pydantic to avoid ambiguity in the order of arguments.

In short, Pydantic validates that incoming data matches the expected type as specified in the type annotation. How is it doing this? Essentially, Pydantic creates a JSON schema for the class, and incoming data is validated against this schema. JSON schmema can take you into a rabbit hole of its own, but the important thing is that it is a JSON representation of the class's attributes, expected types, and some other metadata that can be represented in JSON format. That's going to be important in a moment. But, for what it's worth, here's the JSON schema for the `MyPydanticClass` class:

```json
{
  "title": "MyPydanticClass",
  "type": "object",
  "properties": {
    "value": {
      "title": "Value",
      "type": "integer"
    }
  },
  "required": ["value"]
}
```

In JSON, the keywords "title", "type", etc. are referred to as "fields". Again, that will be important in a moment.

Let's rewrite the Pydantic class with full docstrings:

```python
from pydantic import BaseModel, ValidationError

class MyPydanticClass(BaseModel):
    """The value of the instance. This should be an integer.

    Args:
        value (int): The value to be assigned to the instance.

    Returns:
        None: The instance is created successfully.

    Raises:
        ValidationError: If the value is not an integer or cannot be coerced to an integer.
    """
    value: int
```

This is awesome because mkdocs can read the docstring and generate the API documentation automatically.

However, it is possible to write the class in a way that allows finer control over the JSON schema using Pydantic's `Field` class. The `Field` class allows you to specify additional metadata for the field, such as a default value, a description, and other constraints. This is useful when you want to provide more information about the field or when you want to enforce certain constraints on the field. Here's what it looks like:

```python
from pydantic import BaseModel, Field, ValidationError

class MyPydanticClass(BaseModel):
    value: int = Field(
        ...,
        json_schema_extra={
            "title": "The value of the instance",
            "description": "This should be an integer.",
            "example": 1
        },
    )
```

For reference, the JSON schema's "value" field now looks like this:

```json
{
    "title": "The value of the instance",
    "description": "This should be an integer.",
    "type": "integer",
    "example": 1
}
```

For a human reading the Pydantic class, there is really no need for arguments in a docstring since you can see all the necessary information. Adding this information would be redundant and would clutter the code. As a result, I have not done so wherever I use the Pydantic `Field` class in the Lexos codebase. However, I am not sure if mkdocs can access data from the `Field` class. If it cannot, then we will have to add full docstrings to classes and functions. This needs to be explored further.

Some other features of Pydantic that are both useful and annoying:

```python
from pydantic import validate_call

@validate_call
def print_value(value: int) -> None:
    print(value)

print_value({"value": 1})
```

This will raise a `ValidationError` because the `@validate_call` decorator tells, Pydantic to validate the arguments passed to the function based on the type annotation. It works with any function; you don't need to instantiate a class. Win!

```python
from pydantic import validate_call
import spacy
from spacy.tokens import Doc
nlp = spacy.load("en_core_web_sm")

@validate_call
def print_spacy_doc(doc: Doc) -> str:
    print(doc.text)

doc = nlp("This is a test.")
print_spacy_doc(doc)
```

This will return a `ValidationError` because the spaCy `Doc` class is not recognised in the Pydantic `BaseModel`. Luckily, spaCy also uses Pydantic and has a schema available. So you need to remember to import it and add it to the Pydantic class's configuration:

```python
from pydantic import ConfigDict, validate_call
import spacy
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc
nlp = spacy.load("en_core_web_sm")

config = ConfigDict(json_schema=DocJSONSchema.schema())

@validate_call(config=config)
def print_spacy_doc(doc: Doc) -> str:
    print(doc.text)

doc = nlp("This is a test.")
print_spacy_doc(doc)
```

(If working with a class, you simply add `model_config=config` as a class attribute.) However, not all third-party libraries have importable JSON schemas. For instance, I have not found a way to match the `pd.DataFrame` type, so validating that input data is a dataframe involves writing a custom validator (which is also possible in Pydantic but naturally adds to the codebase). Sometimes this requires a procedural re-think such as making input a dict and having the function convert it to a dataframe.

These complications may occasionally slow development, but, since we don't know what kind of applications may be using Lexos, it seems worthwhile to implement Pydantic validation so that Lexos functions fail as early as possible when input data is not as expected.
