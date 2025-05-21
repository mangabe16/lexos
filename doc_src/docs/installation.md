# Installation

## Installing Python

Lexos requires Python 3.12 or greater. Our development environment is <a href="https://docs.astral.sh/uv/" target="blank">uv</a>, and Lexos should work if Python is installed using that tool.

## Installing the Lexos API

To install the Lexos API, run

```bash
uv add lexos
```

or

```bash
pip install lexos
```

To update to the latest version, use

```bash
uv pip install -U lexos
```

or

```bash
pip install -U lexos
```

This will install the Lexos API and all of its dependencies.

## Downloading the Default Language Model (Required)

Before using Lexos, you will want to install its default language model:

```bash
python -m spacy download xx_sent_ud_sm
```

For information on how Lexos uses language models, see [Tokenizing Texts](tutorial/tokenizing_texts.md).

## Downloading Additional Language Models (Optional)

This is a minimal model that performs sentence and token segmentation for a variety of languages. If you want a model for a specific language, such as English, download it by providing the name of the model:

```bash
python -m spacy download en_core_web_sm
```

If you are working in another language or need a larger language model, you can download instructions for additional models from the <a href="https://spacy.io/models" target="_blank">spaCy models</a> page.
