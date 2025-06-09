# Installation

## Installing Python

Lexos requires Python 3.12 or greater. Our development environment is <a href="https://docs.astral.sh/uv/" target="blank">uv</a>, and Lexos should work in a Python virtua environment created using that tool. If you are using a different Python environment, you can install Lexos using `pip`.

## Installing the Lexos Package

If using uv, run

```bash
uv add lexos
```

Otherwise, you can install Lexos using pip:

```bash
pip install lexos
```

This will install the Lexos API and all of its dependencies.

By default, uv installs the latest version of Lexos. To update to the latest version with pip, use

```bash
pip install -U lexos
```

## Downloading Language Models

Many features of Lexos use language models created for the Python <a href="https://spacy.io/" target="_blank">spaCy</a>spaCy natural language processing library.When you install Lexos, spaCy's multi-language model <code><a href="https://spacy.io/models/xx#xx_sent_ud_sm" target="_blank">xx_sent_ud_sm</a></code> and small English model <code><a href="https://spacy.io/models/en#en_core_web_sm" target="_blank">en_core_web_sm</a></code> are installed. For information on how Lexos uses language models, see [Tokenizing Texts](tutorial/tokenizing_texts.md).

## Downloading Additional Language Models (Optional)

The `xx_sent_ud_sm` model is a minimal model that can be used for sentence and token segmentation in a variety of languages, while the `en_core_web_sm` model is specifically for English text. If you are working in another language or need a larger language, you may need to download additional language models. You can find information on available models on the <a href="https://spacy.io/models" target="_blank">spaCy models</a> page.

To download a models (for instance, the small Chinese model `zh_core_web_sm`), you can run the following commands in your terminal:

```bash
uv run python -m spacy download zh_core_web_sm
```

or, if you are not using uv, you can run:

```bash
python -m spacy download zh_core_web_sm
```

## Verify Installation

To verify that Lexos is installed correctly, you can run the following command in your terminal:

```bash
uv run python -m lexos --version
```

or, if you are not using uv:

```bash
python -m lexos --version
```

If you are using a Jupyter notebook, you can also check the installation by running the following code in a cell:

```python
import lexos
print(lexos.__version__)
```

This should display the version of Lexos that is installed. If you see an error, please check your installation steps or refer to the [Troubleshooting](#troubleshooting) section below.

## Troubleshooting

In this section, we will cover common issues that may arise during the installation of Lexos and how to resolve them. If you encounter any problems, not covered here, please consider reaching out to the Lexos community or checking the <a href="https://github.com/scottkleinman/lexos/issues" target="_blank">GitHub Issues page</a> for assistance.
