# Code Conventions

For a general overview of code conventions for contributors, see the [section in the contributing guide](contributing.md#code-conventions).

1. [Code Compatibility](#code-compatibility)
2. [Auto-Formatting](#auto-formatting)
3. [Linting](#linting)
4. [Documenting code](#documenting-code)
5. [Type Hints](#type-hints)
6. [Formatting Strings](#formatting-strings)
7. [Structuring Logic](#structuring-logic)
8. [Naming](#naming)
9. [I/O and Handling Paths](#io-and-handling-paths)
10. [Error Handling](#error-handling)
11. [Writing Tests](#writing-tests)

## Code Compatibility

Lexos supports **Python 3.12** and above, so all code should be written compatible with 3.12.

## Auto-Formatting

Lexos uses [`ruff`](https://docs.astral.sh/ruff/) for auto-formatting. `ruff` is an opinionated Python code formatter, optimized to produce readable code and small diffs. The following examples show how to run `ruff` from the command-line.

### Lint Your Code

```bash
uv run ruff check .
```

### Auto-Fix Linting Issues

```bash
uv run ruff check . --fix
```

### Format Your Code

```bash
uv run ruff format .
```

You can also run `ruff` in your code editor. For example, if you're using [Visual Studio Code](https://code.visualstudio.com/), you can set up **Ruff** as the default formatter. Add the following to your `settings.json` (in the command pallette, type `Preferences: Open Settings (JSON)`):

```json
{
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "ruff.linting.enabled": true,
    "ruff.linting.run": "onType"
}
```

You may wish to comment out `editor.formatOnSave` if you want to manually format your code. Then install the **Ruff** extension from the VS Code marketplace. You can automatically format your code by running the command palette (`Ctrl+Shift+P`) and selecting "Format Document" or "Format Selection" or by right-clicking and selecting these options.

In some specific cases, e.g. in the tests, it can make sense to disable auto-formatting for a specific block. You can do this by wrapping the code you wish to exclude from formatting in `# fmt: off` and `# fmt: on`.

## Linting

Lexos also uses `ruff` for enforcing code style (linting). It scans one or more files and outputs errors and warnings. This feedback can help you stick to general standards and conventions, and can be very useful for spotting potential mistakes and inconsistencies in your code. Code you write should be compatible with our `ruff` rules and not cause any warnings.

The most common problems surfaced by linting are:

- **Trailing or missing whitespace.** This is related to formatting and should be fixed automatically by running `ruff`.
- **Unused imports.** Those should be removed if the imports aren't actually used. If they're required, e.g. to expose them so they can be imported from the given module, you can add a comment and `# noqa: F401` exception (see details below).
- **Unused variables.** This can often indicate bugs, e.g. a variable that's declared and not correctly passed on or returned. To prevent ambiguity here, your code shouldn't contain unused variables. If you're unpacking a list of tuples and end up with variables you don't need, you can call them `_` to indicate that they're unused.
- **Redefinition of function.** This can also indicate bugs, e.g. a copy-pasted function that you forgot to rename and that now replaces the original function.
- **Repeated dictionary keys.** This either indicates a bug or unnecessary duplication.
- **Comparison with `True`, `False`, `None`**. This is mostly a stylistic thing: when checking whether a value is `True`, `False` or `None`, you should be using `is` instead of `==`. For example, `if value is None`.

To ignore a given line, you can add a comment like `# noqa: F401`, specifying the code of the error or warning we want to ignore. It's also possible to ignore several comma-separated codes at once, e.g. `# noqa: E731,E123`. In general, you should always **specify the code(s)** you want to ignore – otherwise, you may end up missing actual problems. Pull requests containing disabled linting will be considered on a case by case basis.

## Documenting Code

### Docstring Style

All functions and methods you write should be documented with a docstring inline. The docstring can contain a simple summary, and an overview of the arguments and their (simplified) types. Modern editors will show this information to users when they call the function or method in their code.

The Lexos project follows the [Google Style Python Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings. This is a widely used style guide that provides a consistent format for writing docstrings in Python code. It is recommended to follow this style guide for all docstrings in the Lexos project.

!!! note
    The Lexos project uses [mkdocs](https://www.mkdocs.org/) to generate API documentation from directly docstrings. The API documentation is automatically generated from the docstrings in the codebase, so it is important to keep the docstrings up to date and consistent with the code.

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

### Inline Code Comments

We don't expect you to add inline comments for everything you're doing – this should be obvious from reading the code. If it's not, the first thing to check is whether your code can be improved to make it more explicit. That said, if your code includes complex logic or aspects that may be unintuitive at first glance (or even included a subtle bug that you ended up fixing), you should leave a quick comment that provides more context.

```diff
token_index = indices[value]
+ # Index describes Token.i of last token but Span indices are inclusive
span = doc[prev_token_index:token_index + 1]
```

```diff
+ # To create the components we need to use the final interpolated config
+ # so all values are available (if component configs use variables).
+ # Later we replace the component config with the raw config again.
interpolated = filled.interpolate() if not filled.is_interpolated else filled
```

Don't be shy about including comments for tricky parts that _you_ found hard to implement or get right – those may come in handy for the next person working on this code, or even future you!

If your change implements a fix to a specific issue, it can often be helpful to include the issue number in the comment, especially if it's a relatively straightforward adjustment:

```diff
+ # Ensure object is a Span, not a Doc (#1234)
if isinstance(obj, Doc):
    obj = obj[obj.start:obj.end]
```

### Including TODOs

It's fine to include code comments that indicate future TODOs, using the `TODO:` prefix. Modern editors typically format this in a different color, so it's easy to spot. TODOs don't necessarily have to be things that are absolutely critical to fix fight now – those should already be addressed in your pull request once it's ready for review. But they can include notes about potential future improvements.

```diff
+ # TODO: this is currently pretty slow
dir_checksum = hashlib.md5()
for sub_file in sorted(fp for fp in path.rglob("*") if fp.is_file()):
    dir_checksum.update(sub_file.read_bytes())
```

If any of the TODOs you've added are important and should be fixed soon, you should add a GitHub issue that details the task.

## Type Hints

We use Python type hints across the `.py` files wherever possible. This makes it easy to understand what a function expects and returns, and modern editors will be able to show this information to you when you call an annotated function.

If possible, you should always use the more descriptive type hints like `List[str]` or even `List[Any]` instead of only `list`. We also annotate arguments and return types of `Callable` – although, you can simplify this if the type otherwise gets too verbose (e.g. functions that return factories to create callbacks). Remember that `Callable` takes two values: a **list** of the argument type(s) in order, and the return values.

```diff
- def func(some_arg: dict) -> None:
+ def func(some_arg: Dict[str, Any]) -> None:
    ...
```

```python
def create_callback(some_arg: bool) -> Callable[[str, int], List[str]]:
    def callback(arg1: str, arg2: int) -> List[str]:
        ...

    return callback
```

In some cases, you won't be able to import a class from a different module to use it as a type hint because it'd cause circular imports. For instance, `spacy/util.py` includes various helper functions that return an instance of `Language`, but we couldn't import it, because `spacy/language.py` imports `util` itself. In this case, we can provide `"Language"` as a string and make the import conditional on `typing.TYPE_CHECKING` so it only runs when the code is evaluated by a type checker:

```python
from typing TYPE_CHECKING

if TYPE_CHECKING:
    from .language import Language

def load_model(name: str) -> "Language":
    ...
```

Note that we typically put the `from typing` import statements on the first line(s) of the Python module.

## Formatting Strings

Wherever possible, use [f-strings](https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals) for any formatting of strings.

## Structuring Logic

### Positional and Keyword Arguments

We generally try to avoid writing functions and methods with too many arguments, and use keyword-only arguments wherever possible. Python lets you define arguments as keyword-only by separating them with a `, *`. If you're writing functions with additional arguments that customize the behavior, you typically want to make those arguments keyword-only, so their names have to be provided explicitly.

```diff
- def do_something(name: str, validate: bool = False):
+ def do_something(name: str, *, validate: bool = False):
    ...

- do_something("some_name", True)
+ do_something("some_name", validate=True)
```

This makes the function calls easier to read, because it's immediately clear what the additional values mean. It also makes it easier to extend arguments or change their order later on, because you don't end up with any function calls that depend on a specific positional order.

Note that Pydantic enforces the use of keyword arguments instead of positional arguments.

### Avoid Mutable Default Arguments

A common Python gotcha are [mutable default arguments](https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments): if your argument defines a mutable default value like `[]` or `{}` and then goes and mutates it, the default value is created _once_ when the function is created and the same object is then mutated every time the function is called. This can be pretty unintuitive when you first encounter it. We therefore avoid writing logic that does this.

### Don't Use `try`/`except` for Control Flow

We discourage using `try`/`except` blocks for anything that's not third-party error handling or error handling that we otherwise have little control over. There's typically always a way to anticipate the _actual_ problem and **check for it explicitly**, which makes the code easier to follow and understand, and prevents bugs:

```diff
- try:
-     token = doc[i]
- except IndexError:
-     token = doc[-1]

+ if i < len(doc):
+     token = doc[i]
+ else:
+     token = doc[-1]
```

Even if you end up having to check for multiple conditions explicitly, this is still preferred over a catch-all `try`/`except`. It can be very helpful to think about the exact scenarios you need to cover, and what could go wrong at each step, which often leads to better code and fewer bugs. `try/except` blocks can also easily mask _other_ bugs and problems that raise the same errors you're catching, which is obviously bad.

If you have to use `try`/`except`, make sure to only include what's **absolutely necessary** in the `try` block and define the exception(s) explicitly. Otherwise, you may end up masking very different exceptions caused by other bugs.

```diff
- try:
-     value1 = get_some_value()
-     value2 = get_some_other_value()
-     score = external_library.compute_some_score(value1, value2)
- except:
-     score = 0.0

+ value1 = get_some_value()
+ value2 = get_some_other_value()
+ try:
+     score = external_library.compute_some_score(value1, value2)
+ except ValueError:
+     score = 0.0
```

### Avoid Lambda Functions

`lambda` functions can be useful for defining simple anonymous functions in a single line, but they also introduce problems: for instance, they require [additional logic](https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions) in order to be pickled and are pretty ugly to type-annotate. So we typically avoid them in the code base and only use them in the serialization handlers and within tests for simplicity. Instead of `lambda`s, check if your code can be refactored to not need them, or use helper functions instead.

```diff
- split_string: Callable[[str], List[str]] = lambda value: [v.strip() for v in value.split(",")]

+ def split_string(value: str) -> List[str]:
+     return [v.strip() for v in value.split(",")]
```

### Iteration and Comprehensions

We generally avoid using built-in functions like `filter` or `map` in favor of list or generator comprehensions.

```diff
- filtered = filter(lambda x: x in ["foo", "bar"], values)
+ filtered = (x for x in values if x in ["foo", "bar"])
- filtered = list(filter(lambda x: x in ["foo", "bar"], values))
+ filtered = [x for x in values if x in ["foo", "bar"]]

- result = map(lambda x: { x: x in ["foo", "bar"]}, values)
+ result = ({x: x in ["foo", "bar"]} for x in values)
- result = list(map(lambda x: { x: x in ["foo", "bar"]}, values))
+ result = [{x: x in ["foo", "bar"]} for x in values]
```

If your logic is more complex, it's often better to write a loop instead, even if it adds more lines of code in total. The result will be much easier to follow and understand.

```diff
- result = [{"key": key, "scores": {f"{i}": score for i, score in enumerate(scores)}} for key, scores in values]

+ result = []
+ for key, scores in values:
+     scores_dict = {f"{i}": score for i, score in enumerate(scores)}
+     result.append({"key": key, "scores": scores_dict})
```

### Composition vs. Inheritance

Although Lexos uses a lot of classes, **inheritance is viewed with some suspicion** — it's seen as a mechanism of last resort. You should discuss plans to extend the class hierarchy before implementing. Unless you're implementing a new data structure or pipeline component, you typically shouldn't have to use classes at all.

### Don't Use `print`

The core library never `print`s anything. While we encourage using `print` statements for simple debugging (it's the most straightforward way of looking at what's happening), make sure to clean them up once you're ready to submit your pull request. If you want to output warnings or debugging information for users, use the respective dedicated mechanisms for this instead (see sections on warnings and logging for details).

## Naming

Naming is hard. The best we can hope for is if everyone follows some basic conventions.

Consistent with general Python conventions, we use `CamelCase` for class names including dataclasses, `snake_case` for methods, functions and variables, and `UPPER_SNAKE_CASE` for constants, typically defined at the top of a module. We also avoid using variable names that shadow the names of built-in functions, e.g. `input`, `help` or `list`.

### Naming Variables

Variable names should always make it clear _what exactly_ the variable is and what it's used for. Instances of common classes should use the same consistent names. We try to avoid introducing too many temporary variables, as these clutter your namespace. It's okay to re-assign to an existing variable, but only if the value has the same type.

### Naming Methods and Functions

Try choosing short and descriptive names wherever possible and imperative verbs for methods that do something, e.g. `disable_pipes`, `add_patterns` or `get_vector`. Private methods and functions that are not intended to be part of the user-facing API should be prefixed with an underscore `_`. It's often helpful to look at the existing classes for inspiration.

Objects that can be serialized should aim to implement the same consistent methods for serialization. Those usually include at least `to_disk`, `from_disk`, `to_bytes` and `from_bytes`. Some objects can also implement more specific methods like `{to/from}_dict` or `{to/from}_str`.

## I/O and Handling Paths

Code that interacts with the file-system should, if possible accept objects that follow the `pathlib.Path` API. Ideally, user-facing functions and methods should accept `pathlib.Path` objects as input, although in some cases string inputs may be converted to `pathlib.Path` objects early in the function's operation. It is acceptable to convert `pathlib.Path` objects to strings for internal operation.

## Error Handling

We always encourage writing helpful and detailed custom error messages for everything we can anticipate going wrong, and including as much detail as possible. These should be passed to the `LexosException` exception handler.

```python
if something_went_wrong:
    raise LexosException("Something went wrong!")
```

or

```python
try:
    # code that raises a ValueError
except ValueError as e:
    raise LexosException(f"Something went wrong: {e}")
```

The second example exemplifes what we might do if we anticipate possible errors in third-party code that we don't control, or our own code in a very different context, we typically try to provide custom and more specific error messages if possible. This is an example of [re-raising `from`](https://docs.python.org/3/tutorial/errors.html#exception-chaining) the original caught exception so the user sees both the original error, as well as the custom message.

### Avoid Using Naked `assert`

During development, it can sometimes be helpful to add `assert` statements throughout your code to make sure that the values you're working with are what you expect. However, as you clean up your code, those should either be removed or replaced by more explicit error handling:

```diff
- assert score >= 0.0
+ if score < 0.0:
+     raise ValueError(Errors.789.format(score=score))
```

### Warnings

Instead of raising an error, some parts of the code base can raise warnings to notify the user of a potential problem. This is done using Python's `warnings.warn`. Whether or not warnings are shown can be controlled by the user, including custom filters for disabling specific warnings using a regular expression matching our internal codes, e.g. `W123`.

```diff
- print("Warning: No examples provided for validation")
+ warnings.warn(Warnings.W123)
```

When adding warnings, make sure you're not calling `warnings.warn` repeatedly, e.g. in a loop, which will clog up the terminal output. Instead, you can collect the potential problems first and then raise a single warning. If the problem is critical, consider raising an error instead.

```diff
+ n_empty = 0
for spans in lots_of_annotations:
    if len(spans) == 0:
-       warnings.warn(Warnings.456)
+       n_empty += 1
+ warnings.warn(Warnings.456.format(count=n_empty))
```
