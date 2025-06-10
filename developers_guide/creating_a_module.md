# Creating a Module

To create a new module in the Lexos project, you should create a new branch from `main` and follow these steps:

1. **Create a New Directory**: Create a new directory for your module inside the Lexos `src` package. The directory name should be descriptive and follow the naming conventions of the project. If you are creating a submodule of an existing module, simply create the new directory inside the parent module's folder.

2. **Create an `__init__.py` File**: Inside your new module directory, create an `__init__.py` file. This file can be empty or contain initialization code for your module. It is required to make Python treat the directory as a package. The `__init__.py` file should begin with `__init__.py.` (note the period at the end), but it does not need to contain anything else. If you are creating a submodule, you should also create the `__init__.py` file in your submodule's folder.

3. **Create Module Files**: For some simple modules, you can add your code to the `__init__.py` file. You can then import the module with `import lexos.your_module` or `from lexos.your_module import some_function`.

## Handling Exceptions in Your Module

When creating a module, you should handle exceptions properly. If your module raises an exception, it should be a subclass of `lexos.exceptions.LexosException`. This ensures that the exception is consistent with the rest of the Lexos API and can be handled appropriately by users of your module. In general, it is a good idea to add `from lexos.exceptions import LexosException` at the top of your module file to ensure you can raise exceptions correctly.

## Documenting Your Module

Use docstrings to document your module and its functions. This is important for maintaining clarity and usability in the Lexos API. Follow the project's documentation conventions, which can be found in the [Documenting Your Code](documenting_your_code.md) section of the developer guide.

Remember that the content of your type hints and docstrings will be used to generate the documentation for your module. Therefore, it is crucial to ensure that they are clear, concise, and correctly formatted.

## Validating Data with Pydantic

Since we don't know the source of the data that will be passed to your module, it is essential to validate the data before processing it. Lexos uses Pydantic for data validation, which allows you to define data models and ensure that the data conforms to the expected structure. As a general procedure, we try to use Pydantic validation with all public methods and functions in the Lexos API. This helps catch errors early and provides clear feedback to users about the data they are providing. You can find more information about using Pydantic in the [A Short Primer on Pydantic](pydantic_primer.md) section of the developer guide.

!!! Warning
    Pydantic does not accept spaCy docs out of the box. You need to configure it to do so by importing the spaCy `Doc` JSON schema and using Pydantic's `model_config` attribute to pass it to Pydantic. Look at some of the existing modules for examples of how to do this.

## Testing Your Module

It's ideal if you can create tests for your module. To do this, add a new folder for your module in the `uv_lexos/tests` directory, and add your test files there. If you do not have time to write test files, it is acceptable to try out your code in a notebook and save the notebook within your module's directory. This notebook will be picked up later as the basis for the module's documentation and tutorial.
