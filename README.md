# README

The Lexos API is designed to implement many common procedures in the Lexomics toolkit in a way that saves the user having to re-invent the wheel or figure out how to combine multiple Python packages to achieve a given result. It is intended to be used as a library in other projects, but it can also be used as a standalone tool. The API is designed to be modular, so that users can pick and choose the parts they need. The API is also designed to be extensible, so that users can add their own modules. The Lexos API can thus be used in scripts, Jupyter notebooks, or as the back end of a web app.

This project is a major refactor from v0.0.1. The intended release is v0.1.0.

Part of the motivation for this refactor is to adopt `uv` for managing project dependencies. It also bumps the API to Pydantic v2 for type validation and extends its use more regularly across the code base. All classes inherit from Pydantic `BaseModel` and validate public methods with the `@validate_call` decorator (with occasional workarounds for data types which Pydantic does not yet support).

As part of the refactor, the API has been refined and simplified in many different places.

In addition, extensive unit tests have been added to the library. The tests are written in `pytest` and can be run with `uv run pytest`. The tests are designed to be run in a CI/CD pipeline, so that the library can be continuously integrated and deployed.

Note that unit tests have been auto-generated using GitHub Copilot and then tweaked to ensure that they work. However, these tests have their limitations, and it has not yet determined what level of coverage they provide for the entire code base.

## Current Status

As of April 28, 2025, the `io`, `cutter`, `milestones`, `scrubber`, `tokenizer`, `dtm`, `rolling_windows`, `topic_modeling`, `visualization` (but needs some reorganisation and possibly name changes for some classes), and `cluster` are done. (Note that `cluster` is not yet fully implemented, but the `dendrogram` and `clustermap` classes are done, and there is a matplotlib version of `boostrap_consensus`.)

The following modules should be added next (in order of priority):

1. `corpus`
2. `cluster.kmeans`

I have not yet added the API and tutorial documentation.

Other features that need to be be implemented:

1. Similarity Query
2. Content Analysis
3. Statisics

## Setup Guide for Developers

## Setup Procedure

### Step 0: Install `uv`

Instructions for installing `uv` can be found at https://docs.astral.sh/uv/getting-started/installation/. Use the appropriate `curl` or `irm` method for your version to install `uv` universally.

`uv` manages its own virtual environments for each project, so you don't need to install it into a dedicated virtual environment. If it does not find the appropriate version of Python on your system, it will download it automatically.

### Step 1: Clone the Repository

```bash
git clone scottkleinman/uv_lexos
```

`cd` into the project directory and run `uv venv` to create virtual environment. Then run `uv pip sync pyproject.toml`. This will install all dependencies into the `.venv` directory, including the appropriate version of Python.

Activate the environment with `.venv/Scripts/activate` (use backslashes on Windows). You can deactivate the environment with `deactivate`.

To add a dependency, use `uv add <package-name>`. To add a development dependency, use `uv add --dev <package-name>`. To remove a dependency, use `uv remove <package-name>`. To update a dependency, use `uv update <package-name>`. To update all dependencies, use `uv update`.

### Special Considerations

#### spaCy Models

I have not yet figured out how to use `uv` to automate the installation of spaCy models, so you will need to install them manually. Use `uv run python -m spacy download xx_sent_ud_sm` and `uv run python -m spacy download en_core_web_sm`.

One possible method for automatically installing the models would be to put this procedure in a separate script and then call it from `pyproject.toml` file in the `tool.uv.sources` table, as shown below:

```toml
[tool.uv.sources]
setuptools = { path = "./scripts/install_models.py" }
```

The `install_models.py` script would then run `uv run spacy install en_core_web_sm`, etc., which would hopefully install the models in the virtual environment. However, I have not tested this procedure.

Here are some other possible approaches:

Create a script called `download_model.py` with the following contents:

```python
# /// script
# dependencies = [
#   "pip",
# ]
# ///
import spacy
import sys
sys.exec("python -m spacy download xx_sent_ud_sm")
sys.exec("python -m spacy download en_core_web_sm")
```

Then call `uv run download_model.py` to run the script.

It is also possible that `uvx spacy download en_core_web_sm` or `uvx --from spacy download en_core_web_sm` will work.

Either way, at least one extra command has to be run on the command line. This procedure is useful to know if you want to add additional models to the project, but it is preferable if the installation procedure for the default models is invoked from the `pyproject.toml` file.

#### Mimetype Detection

The `io` module uses `python-magic` to detect the mimetype of a file. This library requires the `libmagic` library to be installed on the system. However, the latest version of `python-magic` is unable to install the most recent version of `libmagic`. So the `pyproject.toml` file specifies `v0.4.14`, which is the last compatible version. Hopefully, `python-magic` will be updated to manage this dependency properly.

As a side note, it is worth considering using [https://github.com/cdgriffith/puremagic](https://github.com/cdgriffith/puremagic) instead of `python-magic` for mimetype detection.

### Step 2: Run the Tests

`cd` into the project directory and run `uv run pytest`. This will run all the tests in the `tests` directory. Or you can run `uv run pytest tests/test_module.py` to run the tests in a specific module.
