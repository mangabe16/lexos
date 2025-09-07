# IO

The `IO` module contains the classes and methods useful for loading in texts and text data from various souces and formats into a consistant structure so they can be analyzed within the Lexos enviroment.

This module contains three main components:

1. `BaseLoader`: Central abstract class. Not used directly for loading files, but provides a blueprint and common features for the other loader classes.
2. `Loader`: The main loader used for Lexos. Designed to handle individual files (.txt, .pdf, and docx), directories of files, and zip archives.
3. `DataLoader`: A specialized loader for structured data files such as CSVs, JSON, or Excel files.

## BaseLoader

The `BaseLoader` class is an abstract base class that defines the common interface and functionality for all loader classes. It includes methods for loading files, processing text, and handling errors.

All loaders built on `BaseLoader` have the following attributes for storing loaded data:

- `paths`: File paths or other sources of the loaded texts.
- `mime_types`: MIME types of the loaded items.
- `names`: Names assigned to each loaded text.
- `texts`: The text content of the loaded items.
- `errors`: Any errors encountered during loading.

Additionally loaders will have access to the following properties:

- `records`: Returns a list of dictionaries, with each representing a loaded item with keys such as `name`, `path`, and `mime_type`.
- `data`: Returns a single dictionary containing all of the data stored in the loader.
- `df`: Returns the loaded file records in the form of a Pandas DataFrame.

Common methods available to all loaders include:

- `load_dataset`: Abstract method to be implemented by loaders.
- `dedupe`: Removes duplicate entries from the loaded data and returns a DataFrame with the duplicates removed. The fields to be checked for duplication can be specified.
- `show_duplicates`: Returns a DataFrame containing any duplicates found in the data. Can specify which fields to check for duplicates.
- `reset`: Clear all data from a loader instance. Reset to an empty loader.
- `to_csv`: Save the loaded data to a CSV file.
- `to_excel`: Save the loaded data as an Excel file.
- `to_json`: Save the loaded data to a JSON file.

All Lexos loaders inherit these attributes, properties, and methods from `BaseLoader`, so they can be used in a consistent way. The features inherited from `BaseLoader` will be demonstrated when we look at the `Loader` and `BaseLoader` classes below.

## `Loader`

The `Loader` class is the main loader used in Lexos. It is designed to handle a variety of input formats and sources, including individual text files, directories of files, and zip archives. The `Loader` class inherits from `BaseLoader`, so it has all of the attributes, properties, and methods defined in that class.

The `Loader` class can load files with the following extensions:

- `.txt`: Plain text files.
- `.pdf`: PDF documents.
- `.docx`: Microsoft Word documents.
- `.zip`: Zip archives containing any of the above file types.

The `Loader` class automatically detects the file type based on the file extension and uses the appropriate method to extract the text content. It also handles errors gracefully, logging any issues encountered during loading. The path to the source file can be a local file path or a URL. For multiple files, the path can be a list of file paths or a path to a directory.

```python
from lexos.io.loader import Loader

# Create a Loader instance
loader = Loader()

# Sample texts from various sources
loader.load("path/to/file1.txt")
loader.load(["path/to/file2.txt", "path/to/file3.txt"])
loader.load("path/to/directory_of_files")
loader.load("url/to/file4.txt")
```

Once texts are loaded, they can be accessed through the `texts` attribute or the `df` property, which returns a DataFrame of the loaded records. If there is a problem loading a file, the error will be logged in the `errors` attribute.

By default, the `Loader` class assigns names to loaded texts based on the file name, minus the extension. However, custom names can be provided using the `names` parameter when loading files.

```python
from lexos.io.loader import Loader

# Create a Loader instance
loader = Loader(names=["Doc1", "Doc2"])

# Sample texts
loader.load(["path/to/file1.txt", "path/to/file2.txt"])

print(loader.names)
# ["Doc1", "Doc2"]
```

!!! note
    Names assigned to documents can be useful as labels, especially when generating tabular representations or visualisations of your data.

## `DataLoader`

Collections of texts are frequently stored or distributed in a single file, often with one document per line, or in a structured format like JSON. The `DataLoader` class allows you to load these files directly into a Lexos loader.

### Loading Lineated Text Files

The basic method for loading a file with one document per line is as follows:

```python
# Import the DataLoader class
from lexos.io.data_loader import DataLoader

loader = DataLoader()
loader.load_lineated_text("path/to/file.txt")
```

Note that each document will be named "text001", "text002", "text003", etc. unless you provide a list of document names with the `names` parameter:

```python
# Import the DataLoader class
from lexos.io.data_loader import DataLoader

loader = DataLoader(names=["author1", "author2", "author3"])
loader.load_lineated_text("path/to/file.txt")
```

### Loading CSV and Excel Files

The procedure is similar for CSV and Excel files. However, you must designate which columns contain the document name and text by indicating their headers with the `name_col` and `text_col` parameters.

```python
# Import the DataLoader class
from lexos.io.data_loader import DataLoader

loader = DataLoader()
loader.load_csv("path/to/file.csv", name_col="name", text_col="content")
loader.load_csv("path/to/file.tsv", sep="\t", name_col="name", text_col="content")
loader.load_excel("path/to/file.xlsx", name_col="name", text_col="content")
```

If you are working with a tab-separated file, just use the `sep`, parameter as shown above.

!!! note
    Currently, your file must have headers. Setting the `name_col` and `text_col` by column index is on the roadmap.

### Loading JSON Files

In a JSON-formatted file, each document is a separate object consisting of fields in which the value is referenced by the field's key (e.g. `{"text": "Some text here"}). When loading JSON files, it is necessary to specify the key indicating which field contains the text name and which field contains the text content. This is done with the `name_field` and `text_field` parameters, as shown below:

```python
# Import the DataLoader class
from lexos.io.data_loader import DataLoader

loader = DataLoader()
loader.load_json("path/to/file.json", name_field="name", text_field="content")
```

In standard JSON format, each document is separated by a comma. However, data is frequently formatted with each document separated by a new line, known as JSONL format. If your data is formatted as JSONL, indicate this with the `lines` parameter:

```python
loader.load_json("path/to/file.json", lines=True, name_field="name", text_field="content")
```

### Merging Data into Standard Loaders

Texts loaded from a dataset can be merged into a standard loader with the `Loader.load_dataset` method:

```python
# Create a Dataset instance and load some data
dataset = DataSet()
dataset.load_json("path/to/file.json", name_field="name", text_field="content")

# Create a Loader instance and load a single file
loader.load("path/to/file.txt")

# Merge the dataset into the loader
loader.load_dataset(dataset)
```

## Working with Other Forms of Data

If your data is not in a format that can be loaded with the `Loader` or `DataLoader` classes, it is generally possible to use the Python standard library or third-party tools to load the data into memory and then assign it to an instance of `Loader`.

However, you may wish to create your own loader class (e.g. one that uses an authentication token to access a service) to introduce the logic required for your particular type of data. Custom loaders that inherit from the `BaseLoader` class are welcome as pull requests. If they seem useful to other users, they will be accepted into the main Lexos library.
