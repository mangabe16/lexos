# Loading Texts

## Overview

The Lexos `io` module provides tools for loading texts and datasets into memory for analysis. It includes classes for loading individual text files, structured data files, and datasets containing multiple texts. The main classes are [`Loader`](../../api/io/loader) and [`DataLoader`](../../api/io/data_loader) for structured datasets. The `Loader` class is designed to handle various file formats, including plain text files, PDFs, and Word documents (as well as directories and zip archives containing these formats). `Loader` is able to detect file types  automatically using their <a href="https://en.wikipedia.org/wiki/Media_type" target="_blank">MIME types</a>. The `DataLoader` class is tailored for structured data files like CSV, JSON, and Excel files.

A typical workflow would create a `Loader` object and call `Loader.load()` to load the data from disk or download it from the internet. You can access all loaded texts by calling `Loader.texts`.

!!! note
    It is more efficient simply to use Python's `open()` to load texts into a list _if_ you know the file's encoding. The advantage of the `Loader` class is that it automatically coerces the data to <a href="https://en.wikipedia.org/wiki/UTF-8" target="_blank">UTF-8 encoding</a>, and it allows you to use the same method regardless of the file's format or whether it is on your local machine or downloaded from the internet.

When you use a `Loader`, all your data is stored in memory for use in a Lexos workflow. You can save it to disk, but it is largely up to you to keep track of your data folder(s) and file locations. If you wish to have a more sophisticated system for managing your data, look at [Managing a Corpus](managing_a_corpus.md).

!!! note "Note for Developers"
    Both of `Loader` and `DataLoader` inherit from an abstract [`BaseLoader`](../../api/io/base_loader) abstract class, which defines the common features and methods that all loaders should implement. This allows for a consistent interface if you decide to build a custom loader for a data format not currently handled by the existing loaders.

For practice using the Lexos `io` module, see the <a href="https://scottkleinman.github.io/lexos/tutorials/loading_texts.ipynb" target="_blank">Loading Texts tutorial</a>.

## Using the `Loader` Class

Here is a sample of the code for loading a single text file:

```python
#import Loader
from lexos.io.smart import Loader

# Create the loader and load the data
loader = Loader()
loader.load("myfile.txt)

# Print the first text in the Loader
text = loader.texts[0]
print(text)
```

The `load()` function accepts filepaths, urls, or lists of either. If urls are submitted, the content will be downloaded automatically. Valid formats are `.txt` files, `.docx` files, and `.pdf` files, as well as paths to directories or `.zip` files containing only files of these types.

A `Loader` object has six properties:

- `source`: The filepath or url of the last item added to the `Loader`.
- `names`: A list of the names of all items added to the `Loader`. This will normally be the filenames without the extensions, unless you change them.
- `locations`: A list of the filepaths or urls of all items added to the `Loader`.
- `texts`: A list contain the full text of all the items added to the `Loader`.
- `errors`: A list of filepaths or urls for which loading failed.

As you can see from the example above, each of these properties can be accessed by called `Loader.names`, `Loader.texts`, etc.

You can also iterate through a `Loader` and get the `name`, `location`, and `text` of each item:

```python
for item in loader:
    print(item.name)
    print(item.text)
```

If there is a problem loading a file, the `Loader` will record the error and continue loading the other files. You can access the list of errors with `loader.errors`.

The contents of a `Loader` instance can also be accessed with `loader.records` as a list of dictionaries, where each dictionary contains the keys `name`, `path`, and `mime_type`, or as a Pandas DataFrame (tabular format) with `loader.df`. All the data stored in the `Loader` can be viewed as a single dictionary using the `loader.data` property.

The Lexos `Loader` class also has several methods for manipulating the loaded data:

- `to_csv()`, `to_excel()`, `to_json()` can be used to save the loaded data in CSV, Excel, or JSON file formats. See the API documentation for details on how to use these methods.
- `show_duplicates()` returns a DataFrame showing any duplicate items found in the data, with the option to specify which fields to check for duplicates. `dedupe()` removes duplicate entries from the loaded data and returns a DataFrame with the duplicates removed.
- `reset()` clears all data from the loader instance, resetting it to an empty state.

## Using the `DataLoader`

The DataLoader is designed for importing data sources that are aleady structured. It works well for documents that are organized within a single file or similar set of files.

A "dataset" refers to a collection of documents which are often stored and meant to be accessed from a single file. Lexos has a `DatasetLoader` class designed to work with these data sources. Here is an example in which a single plain text file containing one document per line is loaded.

```python
#import Loader
from lexos.io.data_loader import DataLoader

# Create the loader and load the data
data_loader = DataLoader()
data_loader.load("myfile.txt", labels=["Doc1", "Doc2"])
```

Each line in the file is added to the `data_loader.texts` list. Since the filename cannot be used to generate names for each document (unless you want "myfile1", "myfile2", etc.), you need to supply a list of names using the `labels` parameter. These values will then be accessible in `dataset.names`.

A similar technique can be used to load a CSV file. In this case, we indicate which column should be assigned to the names and which column should be assigned to the text. For instance, if our CSV file has a header with the columns `title` and `text`, we can load it like this using the `name_col` and `text_col` parameters:

```python
data_loader = DataLoader()
data_loader.load_csv("myfile.csv", name_col="title", text_col="content")
```

The `DataLoader` class has a similar `load_excel()` method for loading Excel files. It can also load JSON files using `load_json()` which takes `name_field` and `text_field` parameters to designate which fields should be assigned to the loader's `names` and `texts` attributes. For a complete list of methods and parameters, see the API documentation for the `DataLoader` class.

!!! note
    Under the hood, these methods use the Pandas library's <a href="https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html" target="_blank">read_csv()</a>, <a href="https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html" target="_blank">read_excel()</a>, and <a href="https://pandas.pydata.org/docs/reference/api/pandas.read_json.html" target="_blank">read_json()</a> methods. Any of the keywords accepted by those methods can be passed to the `DataLoader` methods. For example, if you have a tab-separated value file, you can specify `sep="\t"` when loading it.

The `DataLoader` class can also access to the `records`, `data`, and `df` properties described above, as well as the deduping methods, `reset()`, and the `to_csv()`, `to_excel()`, and `to_json()` methods.

### Merging Datasets

Data can be merged from one `DataLoader` instnce into another using the `load_dataset()` method. The following example shows the data from the previously used data_loader_csv being merged into a new blank DataLoader.

```python
# Create a new DataLoader instance and load a CSV file
data_loader1 = DataLoader()
data_loader.load_csv("myfile1.csv", name_col="title", text_col="content")

# Create a new DataLoader instance and load a CSV file
data_loader2 = DataLoader()
data_loader2.load_csv("myfile2.csv", name_col="title", text_col="content")

# Merge data the two `DataLoader` instances
new_data_loader1.load_dataset(existing_data_loader2)
```

!!! note
    The `Loader` class also has a `load_dataset()` method for merging data sets into an existing `Loader` instance. It takes a `DataLoader` instance as an argument and merges it into the given `Loader` object.

The `DatasetLoader.load()` method accepts files, urls, and directories of files in `.txt`, `.csv`, `.tsv`, `.xlsx`, `json`, and `jsonl` format, as well zip archives containing files in those formats. As shown above, `.txt` files must be line-delimited, without a header, and must be accompanied by a list of `labels`.

`.csv`, `.tsv`, and `.xlsx` files must have a header line containing the values `title` and `text`. Lexos will use these columns to assign your documents' `name` and `text` values. If your source file has a different header, you can tell Lexos which headers to use, as in the following example:

```python
dataset_loader.load(
    "myfile.tsv",
    title_col="label",
    text_col="content",
    sep="\t"
)
```

The example above also tells Lexos to use a tab as the separator between columns since the file being loaded is a tab-separated value file. Under the hood, Lexos reads the data with the Pandas library's `read_csv`, `read_excel`, and `read_json` file, and you can pass along any keywords accepted by those methods. The `sep` keyword in the example above is an example.

For JSON-formatted files, use `title_field` and `text_field` to assign which columns should be read by Lexos. If your file is in newline-delimited JSON (JSONL) format, add the parameter `lines=True`.

Once loaded, texts and their metadata can be accessed with the `DatasetLoader.data` property. This is a list of dicts where each document dict has keywords for `title` and `text`. To access the first document's title, you would use `Dataset.data[0]["title"]`. When iterating through the dataset, the `data` property is optional:

```python
for item in dataset:
    print(item["title"])
```

produces the same result as

```python
for item in dataset.data:
    print(item["title"])
```

!!! warning
    Notice that iterating through the `DatasetLoader` requires that you reference keywords of a dict (`item["text"]`, where as the `smart` loader yields an object, allowing you to reference `item.text`. We hope to make this behaviour more consistent in the future.

## The `Dataset` Class

Internally, the `DatasetLoader` detects the format of the input data and then calls the appropriate method of the `Dataset` class. For instance, if the file is a CSV file, the `Dataset.parse_csv()` method will be used. In most case, it makes sense to take advantage of the `DatasetLoader`'s format detection so that you can use the same syntax for all inputs, but in some circumstances, it may be useful to call `Dataset` directly. Here is an example of how you would do it:

```python
from lexos.io.dataset import Dataset

dataset = Dataset.parse_csv("myfile.csv")

for item in dataset:
    print(item["title"])
```

`Dataset.parse_csv()` takes the same `text_col` and `title_col` arguments that you would pass to the `DatasetLoader`. Here is a list of the main `Dataset` methods and the arguments they take:

- `parse_string()`: Parses line-delimited text files. Requires `labels`.
- `parse_csv()`: Parses a CSV file. Requires `text_col` and `title_col` if there are no `text` and `title` headers. Requires `sep="\t"` is the file is a tab-separated value file.
- `parse_excel()`: Parses an Excel file. Requires `text_col` and `title_col` if there are no `text` and `title` headers.
- `parse_json()`: Parses a JSON file. Requires `text_field` and `title_field` if there are no `text` and `title` fields.
- `parse_jsonl()`: Parses a JSONL file. Requires `text_field` and `title_field` if there are no `text` and `title` fields.

## Adding Datasets to a Standard Lexos Loader

If you already have a Loader, it is easy to add datasets to it.

```python
# Import the loaders
from lexos.io.smart import Loader
from lexos.io.dataset import Dataset, DatasetLoader

# Create and empty `Loader`
loader = Loader()

# Create a `DatasetLoader` and load a dataset
dataset_loader = DatasetLoader("myfile1.csv")

# Load a dataset with `Dataset`
dataset = Dataset.parse_csv("myfile1.csv")

# Add the text and names for each dataset to the standard loader
for item in [dataset_loader, dataset]:
    loader.names.extend(item.names)
    loader.texts.extend(item.texts)
```

## What Next?

Once you have all your data in a `Loader`, you can manipulate the text. Almost inevitably, some of the text you have loaded will be "dirty" &mdash; meaning that it is not quite in the shape you want it in for further analysis. This may be a moment to do some preprocessing with the [Scrubber module](user_guide/scrubbing_texts.md).
