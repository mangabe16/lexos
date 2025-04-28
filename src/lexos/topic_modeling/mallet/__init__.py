"""__init__.py.

Last Update: April 27, 2025
Last Tested: April 27, 2025

A fork of Maria Antoniak's Little Mallet Wrapper: https://github.com/maria-antoniak/little-mallet-wrapper.
Here is a rough summary of the changes:

- Some functions for importing training data from various sources.
- Formatting changes, type hinting, and Pydantic validation.
- A more object-oriented approach to keep track of paths and other metadata so that fewer arguments need to be passed to functions.
- Support for a fuller range of MALLET keyword arguments, including the output-state-file which is needed for generating PyLDAVis and Dfr-Browser visualizations.
- Optional progress tracking during training.
- Topic clouds visualisation.
- More parameters for customising the plotting functions.

# TODO:
  - Add topics over time method
  - Because plot figures are not saved to the instance, there can be no helper method to show them in a notebook. Options are: (1) set `show=True` when calling the method; (2) assign the method out to a variable (e.g. `fig = Mallet.plot_method()` and use either `fig` or `display(fig)` in the notebook cell.
"""

import glob
import math
import os
import re
import subprocess
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.figure import Figure
from matplotlib.typing import ColorType
from pydantic import BaseModel, ConfigDict, Field, validate_call
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from spacy.tokens import Doc
from wordcloud import WordCloud

from lexos.exceptions import LexosException

# Get the path to the MALLET binary from the environment
load_dotenv()
MALLET_BINARY_PATH = os.getenv("MALLET_BINARY_PATH")


@validate_call
def import_data_file(file: Path | str) -> list[str]:
    """Import data from a single text file with one document per line.

    Args:
        file (Path | str) A file containing the documents to import.

    Returns:

        list[str]: The training data.
    """
    # Retrieve the data from each file
    try:
        with open(file, "r", encoding="utf-8") as f:
            training_data = f.readlines()
    except FileNotFoundError:
        raise LexosException(f"File {file} does not exist.")
    except IOError:
        raise LexosException(f"File {file} could not be read.")

    return training_data


@validate_call
def import_dirs(dirs: Path | str | list[Path | str]) -> list[str]:
    """Import a directory or list of directories.

    Args:
        dirs (Path | str | list[Path | str]) A directory or list of directories to import.

    Returns:

        list[str]: The training data.
    """
    # Ensure dirs is a list
    if isinstance(dirs, (Path, str)):
        dirs = [dirs]
    # Retrieve file paths or raise an error if the directory does not exist
    training_data = []
    for dir in dirs:
        if not Path(dir).is_dir():
            raise LexosException(f"Directory {dir} does not exist.")
        else:
            # NOTE: Cannot use Path.glob() here because it returns a generator, which disrupts testing.
            filepaths = glob.glob(f"{dir}/*.txt")
            for path in filepaths:
                if Path(path).is_file():
                    with open(path, "r", encoding="utf-8") as f:
                        training_data.append(f.read())

    return training_data


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def import_docs(docs: list[str | Doc]) -> list[str]:
    """Import a list of documents.

    Args:
        docs (list[str | Doc]) List of documents to import. Each document can be a string or a Doc object.

    Returns:

        list[str]: The training data.

    """
    training_data = []
    for doc in docs:
        if isinstance(doc, Doc):
            training_data.append(doc.text)
        else:
            training_data.append(doc)

    return training_data


@validate_call
def import_files(files: Path | str | list[Path | str]) -> list[str]:
    """Import a directory or list of directories.

    Args:
        files (Path | str | list[Path | str]) A file or list of directories to import.

    Returns:

        list[str]: The training data.
    """
    # Ensure dirs is a list
    if isinstance(files, (Path, str)):
        files = [files]

    # Retrieve the data from each file
    training_data = []
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                training_data.append(f.read())
        except FileNotFoundError:
            raise LexosException(f"File {file} does not exist.")
        except IOError:
            raise LexosException(f"File {file} could not be read.")

    return training_data


class Mallet(BaseModel):
    path_to_mallet: str = Field(
        MALLET_BINARY_PATH,
        json_schema_extra={"description": "The path to the MALLET binary file."},
    )
    metadata: dict[str, Any] = Field(
        {},
        json_schema_extra={
            "description": "A dict containing metadata generated by the class instance."
        },
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def distributions(self) -> list[str]:
        """Get the topic distributions of the model."""
        if "path_to_topic_distributions" not in self.metadata:
            raise LexosException(
                "No topic distributions have been set. Please designate a path for `path_to_topic_distributions` when you train your topic model."
            )

        topic_distributions = []
        with open(self.metadata["path_to_topic_distributions"], "r") as f:
            for line in f.readlines():
                if line.split()[0] != "#doc":
                    try:
                        _, distribution = (line.split("\t")[1], line.split("\t")[2:])
                    except IndexError:
                        raise LexosException(
                            "The line {line} in the topic distributions file is not formatted correctly."
                        )
                    distribution = [float(p) for p in distribution]
                    topic_distributions.append(distribution)

        return topic_distributions

    @property
    def num_docs(self) -> int:
        """Get the number of docs in the model."""
        if "num_docs" in self.metadata:
            return self.metadata["num_docs"]
        else:
            return 0

    @property
    def mean_num_tokens(self) -> int:
        """Get the mean number of tokens per document in the model."""
        if "mean_num_tokens" in self.metadata:
            return self.metadata["mean_num_tokens"]
        else:
            return 0

    @property
    def model_directory(self) -> str:
        """Get the model directory."""
        if "model_directory" not in self.metadata:
            raise LexosException(
                "No model directory has been set. A directory is created by default when you call `import_data()`"
            )
        return self.metadata["model_directory"]

    @cached_property
    def topic_keys(self) -> list[list[str]]:
        """Get the keys of the model.

        Returns:
            list[list[str]]: A list of topics where each topic is a sublist containing the topic index, topic weight, and a space-separated list of keywords.
        """
        if "path_to_topic_keys" not in self.metadata:
            raise LexosException(
                "No topic keys have been set. Please designate a path for `path_to_topic_keys` when you train your topic model."
            )

        with open(self.metadata["path_to_topic_keys"], "r") as f:
            return [line.strip().split("\t") for line in f.readlines()]

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of documents in the model."""
        if "vocab_size" in self.metadata:
            return self.metadata["vocab_size"]
        else:
            return 0

    def _setup_wordcloud(self, round_mask, max_terms, **kwargs) -> WordCloud:
        """Set up the word cloud object.

        Args:
            round_mask (bool): Whether to use a round mask for the word cloud.
            max_terms (int): The maximum number of keywords to display.
            **kwargs: Additional keyword arguments for the WordCloud object.

        Returns:
            WordCloud: A configured WordCloud object.
        """
        # Define a mask to make the word cloud round (just some eye candy)
        if round_mask:
            x, y = np.ogrid[:300, :300]
            mask = (x - 150) ** 2 + (y - 150) ** 2 > 130**2
            mask = 255 * mask.astype(int)
        else:
            mask = None

        # Configure the word cloud object
        options = {
            "background_color": "white",
            "mask": mask,
            "contour_width": 0.1,
            "contour_color": "white",
            "max_words": max_terms,
            "min_font_size": 10,
            "max_font_size": 150,
            "random_state": 42,
            "colormap": "Dark2",
        }
        for k, v in kwargs.items():
            options[k] = v

        return WordCloud(**options)

    def _track_progress(
        self, mallet_cmd: str, num_iterations: int, verbose: bool = True
    ) -> None:
        """Track the progress of the modeling.

        Args:
            mallet_cmd (str): The MALLET command to run.
            num_iterations (int): The number of iterations for the model.
            verbose (bool): Whether to print the MALLET output.

        Notes:
            - Prints MALLET output and updates the progress bar in 10% increments.
        """
        console = Console()
        # NOTE: This is a hack to make Jupyter notebooks in VS Code display all lines
        # in the same cell. It may cause undesirable results in other environments and
        # needs further testing. See https://github.com/Textualize/rich/issues/3483.
        if verbose:
            console.is_jupyter = False

        # Create a progress display with rich
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            # Create a task with a total of 100 (percentage)
            task = progress.add_task("[blue]Training model...", total=100)

            # Run the MALLET command
            p = subprocess.Popen(
                mallet_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
            )

            # Regex to match progress information
            prog = re.compile(r"\<([^\)]+)\>")

            # Track the last reported progress percentage to avoid duplicate updates
            last_progress = -1

            # Process the output line by line
            while p.poll() is None:
                line = p.stdout.readline().decode()
                if verbose:
                    # Print MALLET output without disrupting progress
                    console.print(line, end="")

                # Keep track of modeling progress
                try:
                    # A float indicating the percentage, which is output by MALLET
                    this_iter = float(prog.match(line).groups()[0])
                    current_progress = int(100.0 * this_iter / num_iterations)

                    # Only update on 10% multiples and avoid duplicate updates
                    if current_progress % 10 == 0 and current_progress > last_progress:
                        # Update to the current progress percentage
                        progress.update(task, completed=this_iter)
                        last_progress = current_progress
                    if current_progress == 100:
                        progress.update(task, description="[green]Complete", completed=100)
                except AttributeError:  # Not every line will match.
                    pass

    @validate_call(config=model_config)
    def get_keys(
        self, num_topics: int = None, topics: list[int] = None, num_keys: int = 10
    ) -> str:
        """Get a string representation of the topic keys of the model.

        Args:
            num_topics (int): The number of topics to get keys for. If None, get keys for all topics.
            topics (list[int]): A list of topic indices to get keys for. If None, get keys for all topics.
            num_keys (int): The number of keys to output for each topic.

        Returns:
            str: A string representation of the topic keys.
        """
        if num_topics and not topics:
            topic_keys = self.topic_keys[:num_topics]
        elif topics:
            topic_keys = [self.topic_keys[i] for i in topics]
        else:
            topic_keys = self.topic_keys
        output = ""
        for topic in topic_keys:
            keywords = " ".join(topic[2].split()[:num_keys])
            output += f"Topic {topic[0]}\t{topic[1]}\t{keywords}\n"
        return output

    @validate_call(config=model_config)
    def get_top_docs(
        self, topic=0, n=10, metadata: pd.DataFrame = None, as_str: bool = False
    ) -> pd.DataFrame | str:
        """Get the top n documents for a given topic.

        Args:
            topic (int): Topic number.
            n (int): Number of top documents to return.
            metadata (pd.DataFrame): Dataframe with the metadata in the same order as the training data (optional).
            as_str (bool): Whether to return the result as a string instead of a dataframe.

        Returns:
            A pd.DataFrame or str: A dataframe with the top n documents for the given topic, or a string representation of the dataframe.

        Notes:
            - The metadata must be in the same order as the training data.
            - The document text will get ellided by the maximum width of a pandas column. An easy way to see the full text is to set `as_str=True` and output the result with a print statement. You can also use the pandas API to extract the information with something like `top_docs.Document.tolist()`.
        """
        if "path_to_topic_distributions" not in self.metadata:
            raise LexosException(
                "No topic distributions have been set. Please designate a path for `path_to_topic_distributions` when you train your topic model."
            )

        if "path_to_training_data" not in self.metadata:
            raise LexosException(
                "No training data has been set. Please designate a path for `path_to_training_data` when you train your topic model."
            )

        # Read the training data file
        with open(self.metadata["path_to_training_data"], "r", encoding="utf-8") as f:
            training_data = f.readlines()
        training_data = [
            line.split("\t")[2].strip() for line in training_data
        ]  # Skip the id and label

        # Combine the distribution and training data, then convert to a dataframe
        distribution_data = [
            (_distribution[topic], _document)
            for _distribution, _document in zip(self.distributions, training_data)
        ]
        df = pd.DataFrame(distribution_data, columns=["Distribution", "Document"])
        df.index.name = "Doc ID"

        # If metadata is provided, concatenate it to the dataframe
        if metadata is not None:
            df = pd.concat([df, metadata], axis=1)

        # Sort the dataframe by distribution and return the top n documents
        if as_str:
            return (
                df.sort_values(by="Distribution", ascending=False).head(n).to_string()
            )
        return df.sort_values(by="Distribution", ascending=False).head(n)

    @validate_call(config=model_config)
    def get_topic_term_probabilities(
        self, topics: Optional[int | list[int]] = None, n: int = 5
    ) -> str:
        """Get a string representation of the term distribution for a given topic.

        Args:
            topics (int | list[int]): Topic number. If None, get the probabilities for all topics.
            n (int): The number of keywords to display.

        Returns:
            str: A string representation of the term distribution for the given topic.
        """
        if isinstance(topics, int):
            topics = [topics]
        topic_term_probability_dict = self.load_topic_term_distributions()
        result = ""
        for _topic, _term_probability_dict in topic_term_probability_dict.items():
            if topics is None or _topic in topics:
                result += f"Topic {_topic}\n"
                for _term, _probability in sorted(
                    _term_probability_dict.items(), key=lambda x: x[1], reverse=True
                )[:n]:
                    result += f"\t{_term}: {_probability}\n"
                result += "\n"
        return result

    @validate_call(config=model_config)
    def import_data(
        self,
        training_data: list[str],
        path_to_training_data: str,
        path_to_formatted_training_data: Optional[str] = None,
        training_ids: Optional[list[int]] = None,
        use_pipe_from: Optional[str] = None,
    ) -> None:
        """Format the training data for MALLET.

        Args:
            training_data (list[str]): The training data to format.
            path_to_training_data (str): The path to save the training data file.
            path_to_formatted_training_data (Optional[str]): The path to save the formatted data file. By default, the data will be saved in the same directory as the training data file with the name "formatted_training_data.mallet".
            training_ids: Optional[list[int]]: A list of document ids designating a subset of the entire data set. If None, the entire dataset will be imported.
            use_pipe_from: Optional[str]: The path to a pipe from which to read the training data. If None, the training data will be read from the file specified by `path_to_training_data`.
        """
        # Save the training data file in the model directory or use the provided path to create a new one
        if "model_directory" not in self.metadata:
            self.metadata["model_directory"] = Path(path_to_training_data).parent.as_posix()
        else:
            model_dir = Path(self.metadata["model_directory"])
            model_dir.mkdir(exist_ok=True)
            path_to_training_data = (
                f"{self.metadata['model_directory']}/{path_to_training_data}"
            )

        # Get the parent directory of the training data file
        if not path_to_formatted_training_data:
            model_dir = Path(self.metadata["model_directory"])
            path_to_formatted_training_data = (
                model_dir / "formatted_training_data.mallet"
            )

        # Save the training data file
        training_data_file = open(path_to_training_data, "w", encoding="utf-8")
        for i, doc in enumerate(training_data):
            # Remove newlines and carriage returns from the document
            doc = re.sub("[\r\n]+", " ", doc).strip()
            if training_ids:
                training_data_file.write(f"{training_ids[i]}\tno_label\t{doc}\n")
            else:
                training_data_file.write(f"{i}\tno_label\t {doc}\n")
        training_data_file.close()
        self.metadata["path_to_training_data"] = path_to_training_data
        self.metadata["path_to_formatted_training_data"] = (
            path_to_formatted_training_data
        )
        self.metadata["num_docs"] = len(training_data)
        # WARNING: Tokenisation relies on whitespace, so it may not be accurate for all languages
        self.metadata["mean_num_tokens"] = np.mean(
            [len(doc.split()) for doc in training_data]
        )
        self.metadata["vocab_size"] = len(
            list(set([token for doc in training_data for token in doc.split()]))
        )

        # Build and execute the command to format the training data for MALLET
        cmd = f"{self.path_to_mallet} import-file --input {path_to_training_data} --output {path_to_formatted_training_data} --keep-sequence --preserve-case"
        if use_pipe_from:
            cmd += f" --use-pipe-from {use_pipe_from}"
        os.system(cmd)

    def load_topic_term_distributions(self) -> dict[str, float]:
        """Load the topic-term distributions from a file.

        Returns:
            dict[str, float]: A dictionary of all topic-term distributions.
        """
        # Ensure that the path to a term weights file has been set
        if "path_to_term_weights" not in self.metadata:
            raise LexosException(
                "No term weights been set. Please designate a path for `path_to_term_weights` when you train your topic model."
            )
        term_weight_path = self.metadata["path_to_term_weights"]
        topic_term_weight_dict = defaultdict(lambda: defaultdict(float))
        topic_sum_dict = defaultdict(float)
        with open(term_weight_path, "r") as f:
            for _line in f:
                _topic, _term, _weight = _line.split("\t")
                topic_term_weight_dict[_topic][_term] = float(_weight)
                topic_sum_dict[_topic] += float(_weight)

        topic_term_probability_dict = defaultdict(lambda: defaultdict(float))
        for _topic, _term_weight_dict in topic_term_weight_dict.items():
            for _term, _weight in _term_weight_dict.items():
                topic_term_probability_dict[int(_topic)][_term] = (
                    _weight / topic_sum_dict[_topic]
                )

        return topic_term_probability_dict

    @validate_call(config=model_config)
    def plot_categories_by_topic_boxplots(
        self,
        categories: list[str],
        topics: Optional[int | list[int]] = None,
        output_path: Optional[str] = None,
        target_labels: Optional[list[str]] = None,
        num_keys: int = 5,
        figsize: Optional[tuple[int, int]] = (6, 6),
        font_scale: Optional[float] = 1.2,
        color: Optional[ColorType] = "lightblue",
        show: Optional[bool] = True,
    ) -> Figure | list[Figure]:
        """Plot boxplots showing the distribution of topic probabilities for each category.

        Args:
            categories (list[str]): The labels to use for the categories.
            topics (int | list[int]): The index of the topic to plot.
            output_path (str): The path to save the figure.
            target_labels (list[str]): Unique labels for categories to classify.
            num_keys (int): The number of keywords to display.
            figsize: (Optional[tuple[int, int]]): The dimensions of the figure.
            font_scale (Optional[float]): The font scale for the figure.
            color (Optional[ColorType]): The color to use for the heatmap boxes. A matplotlib ColorType name or object.
            show (Optional[bool]): Whether to show the figure.

        Returns:
            Figure | list[Figure]: The boxplot showing the topic associations by category.
        """
        # Load topic_keys
        topic_keys = self.topic_keys

        # Ensure that topics is a list
        if topics is None:
            topics = list(range(len(topic_keys)))
        elif isinstance(topics, int):
            topics = [topics]

        # Ensure there are topic_labels
        if not target_labels:
            target_labels = list(set(categories))

        # Combine the labels and distributions into a dataframe.
        figs = []
        for topic in topics:
            keywords = " ".join(topic_keys[topic][2].split()[:num_keys])

            dicts_to_plot = []
            for _label, _distribution in zip(categories, self.distributions):
                if not target_labels or _label in target_labels:
                    dicts_to_plot.append(
                        {
                            "Probability": float(_distribution[topic]),
                            "Category": _label,
                            "Topic": keywords,
                        }
                    )
            df_to_plot = pd.DataFrame(dicts_to_plot)

            # Show the final plot
            if figsize:
                plt.figure(figsize=figsize)
            sns.set_theme(style="ticks", font_scale=font_scale)
            sns.boxplot(data=df_to_plot, x="Category", y="Probability", color=color)
            sns.despine()
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Topic {topic}: {keywords}")
            plt.tight_layout()
            if output_path:
                plt.savefig(output_path)
            if show:
                plt.show()
                return None
            else:
                figs.append(plt.gcf())
                plt.close()
                if len(figs) == 0:
                    return figs[0]
        return figs

    @validate_call(config=model_config)
    def plot_categories_by_topics_heatmap(
        self,
        categories: list[str],
        output_path: Path | str = None,
        target_labels: list[str] = None,
        num_keys: int = 5,
        figsize: Optional[tuple[int, int]] = None,
        font_scale: Optional[float] = 1.2,
        cmap: Optional[ColorType] = sns.cm.rocket_r,
        show: Optional[bool] = True,
    ) -> Figure:
        """Plot heatmap showing topics by category.

        Args:
            categories (list[str]): The categories to use to classify topics.
            output_path (Path | str): The path to save the figure.
            target_labels (list[str]): Unique labels for categories to classify.
            num_keys (int): The number of keywords to display.
            figsize: (Optional[tuple[int, int]]): The dimensions of the figure.
            font_scale (Optional[float]): The font scale for the figure.
            cmap (Optional[ColorType]): The colormap to use for the heatmap. A matplotlib colormap name or object, or list of colors.
            show (Optional[bool]): Whether to show the figure.

        Returns:
            Figure: The heatmap showing the topic associations by category.
        """
        # Load topic_keys
        topic_keys = self.topic_keys

        # Combine the labels and distributions into a list of dictionaries.
        dicts_to_plot = []
        for _category_label, _distribution in zip(categories, self.distributions):
            if not target_labels or _category_label in target_labels:
                for _topic, _probability in enumerate(_distribution):
                    keywords = " ".join(topic_keys[_topic][2].split()[:num_keys])
                    _topic_label = f"Topic {_topic:02}: {keywords}"
                    dicts_to_plot.append(
                        {
                            "Probability": float(_probability),
                            "Category": _category_label,
                            "Topic": _topic_label,
                        }
                    )

        # Create a dataframe, format it for the heatmap function, and normalize the columns.
        df_to_plot = pd.DataFrame(dicts_to_plot)
        df_wide = df_to_plot.pivot_table(
            index="Category", columns="Topic", values="Probability"
        )
        df_norm_col = (df_wide - df_wide.mean()) / df_wide.std()

        # Show the final plot
        fig, _ = plt.subplots()
        if figsize:
            plt.figure(figsize=figsize)
        sns.set_theme(style="ticks", font_scale=font_scale)
        ax = sns.heatmap(df_norm_col, cmap=cmap)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        plt.xticks(rotation=30, ha="left")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
        if show:
            plt.show()
            return None
        else:
            plt.close()
            return fig

    @validate_call(config=model_config)
    def topic_clouds(
        self,
        topics: Optional[int | list[int]] = None,
        max_terms: Optional[int] = 30,
        figsize: Optional[tuple[int, int]] = (10, 10),
        output_path: Optional[str] = None,
        show: Optional[bool] = True,
        round_mask: Optional[bool] = True,
        layout: Optional[str | tuple[int, int]] = "auto",
        **kwargs,
    ) -> Figure:
        """Get a word cloud representation of the term distribution for a given topic.

        Args:
            topics (int | list[int]): Topic number. If None, get the probabilities for all topics.
            max_terms (int): The maxium number of keywords to display.
            figsize (tuple[int, int]): The size of the word cloud figure(s).
            output_path (Optional[str]): The path to save the figure.
            show (Optional[bool]): Whether to show the figure.
            round_mask (Optional[bool]): Whether to use a round mask for the word cloud.
            layout (Optional[str | tuple[int, int]]): The number of rows and columns in the figure. Default is "auto".
            **kwargs: Additional keyword arguments for the WordCloud object.

        Returns:
            Figure: A pyplot figure with the word clouds.
        """
        # Ensure that topics is a list
        if isinstance(topics, int):
            topics = [topics]

        # Set parameters for plotting
        sns.set_theme()
        plt.rcParams["figure.figsize"] = figsize

        # Configure the WordCloud object
        wordcloud = self._setup_wordcloud(
            round_mask=round_mask, max_terms=max_terms, **kwargs
        )

        # Load the topic_term_probability_dict
        topic_term_probability_dict = self.load_topic_term_distributions()

        # Auto-grid
        if layout == "auto":
            if topics is None:
                n = len(topic_term_probability_dict)
            else:
                n = len(topics)
            columns = math.floor(math.sqrt(n))
            rows = math.ceil(n / columns)
        elif isinstance(layout, tuple):
            rows, columns = layout

        # Generate the word clouds
        for _topic, _term_probability_dict in topic_term_probability_dict.items():
            if topics is None or _topic in topics:
                wordcloud.generate_from_frequencies(_term_probability_dict)
                plt.subplot(rows, columns, _topic + 1)
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"Topic {_topic}")
        if output_path:
            plt.savefig(output_path)
        if show:
            plt.show()
            return None
        else:
            fig = plt.gcf()
            plt.close()
            return fig

    @validate_call(config=model_config)
    def train(
        self,
        num_topics: int,
        path_to_formatted_training_data: Optional[str] = None,
        num_iterations: Optional[int] = 100,
        path_to_model: Optional[str] = None,
        path_to_state: Optional[str] = None,
        path_to_topic_keys: Optional[str] = None,
        path_to_topic_distributions: Optional[str] = None,
        path_to_term_weights: Optional[str] = None,
        path_to_diagnostics: Optional[str] = None,
        optimize_interval: Optional[int] = 10,
        verbose: Optional[bool] = True,
    ) -> None:
        """Train the topic model using MALLET.

        Args:
            num_topics (int): The number of topics to train.
            path_to_formatted_training_data (str): The path to the formatted training data file.
            num_iterations (int): The number of iterations to train for.
            path_to_model (str): The path to save the model file.
            path_to_state (str): The path to save the state file.
            path_to_topic_keys (str): The path to save the topic keys file.
            path_to_topic_distributions (str): The path to save the topic distributions file.
            path_to_term_weights (str): The path to save the term weights file.
            path_to_diagnostics (str): The path to save the diagnostics file.
            optimize_interval (int): The interval at which to optimize the model.
            verbose (bool): Whether to print the MALLET output.
        """
        # Ensure that the model directory is set
        if "model_directory" not in self.metadata:
            raise LexosException(
                "No model directory has been set. Please set a path for `model_directory` in the instance `metadata` dict."
            )

        # Replace the instance path to formatted training data if a new one is provided
        if path_to_formatted_training_data:
            self.metadata["path_to_formatted_training_data"] = (
                path_to_formatted_training_data
            )

        # Check that the path to formatted training data has been set
        if not self.metadata.get("path_to_formatted_training_data"):
            raise LexosException(
                "No training data has been set. Please designate a path for `path_to_formatted_training_data`."
            )

        # Build the MALLET command
        cmd = f"{self.path_to_mallet} train-topics"
        flags = {
            "input": self.metadata["path_to_formatted_training_data"],
            "num-topics": num_topics,
            "num-iterations": num_iterations,
            "inferencer-filename": path_to_model,
            "output-state": path_to_state,
            "output-topic-keys": path_to_topic_keys,
            "output-doc-topics": path_to_topic_distributions,
            "topic-word-weights-file": path_to_term_weights,
            "diagnostics-file": path_to_diagnostics,
            "optimize-interval": optimize_interval,
        }
        for k, v in flags.items():
            if v:
                # Save file names in the model directory if they are not absolute paths
                if isinstance(v, str) and len(Path(v).parts) == 1:
                    v = f"{self.metadata['model_directory']}/{v}"
                cmd += f" --{k} {v}"

        # Train the model
        # print("Training topics...")
        self._track_progress(cmd, num_iterations, verbose)
        self.metadata["num_topics"] = num_topics
        self.metadata["optimize_interval"] = optimize_interval
        # Update the paths
        paths = {
            "path_to_model": path_to_model,
            "path_to_state": path_to_state,
            "path_to_topic_keys": path_to_topic_keys,
            "path_to_topic_distributions": path_to_topic_distributions,
            "path_to_term_weights": path_to_term_weights,
            "path_to_diagnostics": path_to_diagnostics,
        }
        for k, v in paths.items():
            if v and len(Path(v).parts) == 1:
                v = f"{self.metadata['model_directory']}/{v}"
            self.metadata[k] = v
        self.metadata["training_command"] = cmd
        # print("Complete")
