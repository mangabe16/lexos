"""__init__.py.

Last Update: March 9, 2025
Last Tested: March 9, 2025
"""

import glob
import os
import re
import shlex
from collections import Counter
from pathlib import Path
from subprocess import PIPE, STDOUT, CalledProcessError, Popen, check_output
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token
from wasabi import Printer

from lexos.exceptions import LexosException
from lexos.topic_modeling.mallet import scale_model

# Get the path to the MALLET binary from the environment
load_dotenv()
MALLET_BINARY_PATH = os.getenv("MALLET_BINARY_PATH")


# Helper Files
def ensure_posix(path: str) -> str:
    """Ensure a path is in posix format.

    Args:
        path (str): The path to convert.

    Returns:
        str: The path in posix format.
    """
    return Path(path).as_posix()


class Mallet(BaseModel):
    """A class for working with MALLET."""

    model_dir: Optional[str] = Field(
        ...,
        json_schema_extra={"description": "The directory to store the model files."},
    )
    data_file: Optional[str] = Field(
        None, json_schema_extra={"description": "The path to the data file to import."}
    )
    mallet_import_file: Optional[str] = Field(
        None, json_schema_extra={"description": "The path to the MALLET import file."}
    )
    msg: Printer = Field(
        Printer(), json_schema_extra={"description": "A wasabi printer for messages."}
    )
    mallet_bin: str = Field(
        MALLET_BINARY_PATH,
        json_schema_extra={"description": "The path to the MALLET binary."},
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema=DocJSONSchema.schema()
    )

    def __init__(self, **data):
        """Initialize the Import class."""
        super().__init__(**data)

        # Start the message printer
        self.msg = Printer()

        # Ensure all paths are posix for compatibility with subprocess
        self.model_dir = ensure_posix(self.model_dir)
        self.mallet_bin = ensure_posix(self.mallet_bin)
        if not self.data_file:
            self.data_file = f"{self.model_dir}/data.txt"
        else:
            self.data_file = ensure_posix(self.data_file)

    def _get_token_bag(
        self,
        doc: Doc | Span,
        use_lemmas: bool = False,
        remove_stops: bool = False,
        remove_punct: bool = False,
    ) -> str:
        """Get the token bag for a spaCy doc.

        Args:
            doc (Doc | Span | str): A spaCy document or string.
            use_lemmas (bool): Whether to use lemmas.
            remove_stops (bool): Whether to remove stop words.
            remove_punct (bool): Whether to remove punctuation.

        Returns:
            str: A space-separated string of tokens.
        """
        # Filter tokens in a single pass
        tokens = []
        for token in doc:
            if not token.has_extension("is_allowed") or not token._.is_allowed:
                continue

            # Skip stop words if removal is requested
            if remove_stops and token.is_stop:
                continue

            # Skip punctuation if removal is requested
            if remove_punct and token.is_punct:
                continue

            # Append the text or lemma to the list
            if use_lemmas:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

        # Count the terms
        counts = dict(Counter(tokens))

        # Create a bag with copies of each token occurring multiple times
        bag = []
        for k, v in counts.items():
            repeated = f"{k} " * v
            bag.append(repeated.strip())

        return " ".join(bag)

    def _import_data(
        self,
        input: str,
        output: str,
        keep_sequence: bool = False,
        preserve_case: bool = False,
        token_regex: str = "\\p{L}[\\p{L}\\p{P}]+\\p{L}",
    ):
        """Import data from a data file.

        Args:
            input (str): The path to the data file to import.
            output (str): The path to the MALLET import file.
            keep_sequence (bool): Whether to keep the sequence of the data.
            preserve_case (bool): Whether to preserve the case of the data.
            token_regex (str): The regex pattern MALLet will use for tokenization.
        """
        # Ensure path to data source is set and is in posix format
        if input is not None:
            self.data_file = ensure_posix(input)
        elif input is None and self.data_file is None:
            self.msg.fail("Please provide an input file.")

        # Ensure path to import file is set and is in posix format
        if output is not None:
            self.mallet_import_file = ensure_posix(output)
        elif output is None and self.mallet_import_file is None:
            self.mallet_import_file = f"{self.model_dir}/import.mallet"

        # Build the MALLET command
        mallet_cmd = f"{self.mallet_bin} import-file --input {self.data_file} --output {self.mallet_import_file} --keep-sequence {keep_sequence} --preserve-case {preserve_case} --token-regex '{token_regex}'"
        self.msg.text(f"Running {mallet_cmd}...")
        mallet_cmd = shlex.split(mallet_cmd)

        # Perform the import
        # TODO: Possibly avoid check_output since it is really hard to
        # identify a CalledProcessError.
        try:
            # shell=True required to handle backslashes in token-regex
            _ = check_output(
                mallet_cmd, stderr=STDOUT, shell=True, universal_newlines=True
            )
            self.msg.good(f"Import complete. Data imported to {output}.")
        except CalledProcessError as e:
            self.msg.fail(e.output)

    def _set_training_options(
        self,
        import_file,
        num_topics,
        num_iterations,
        optimize_interval,
        random_seed,
        **kwargs,
    ) -> dict[str, Any]:
        """Set the training options.

        Args:
            import_file (str): The path to the MALLET import file.
            num_topics (int): The number of topics to model.
            num_iterations (int): The number of iterations to run.
            optimize_interval (int): The interval to optimize the model.
            random_seed (int): The random seed for the model.
            **kwargs: Additional options for the model.

        Returns:
            dict[str, Any]: The options for the model.
        """
        opts = {
            "input": import_file,
            "num-topics": num_topics,
            "num-iterations": num_iterations,
            "optimize-interval": optimize_interval,
            "random-seed": random_seed,
            "output-state": f"{self.model_dir}/state.gz",
            "output-topic-keys": f"{self.model_dir}/keys.txt",
            "output-doc-topics": f"{self.model_dir}/composition.txt",
            "word-topic-counts-file": f"{self.model_dir}/counts.txt",
            "output-topic-docs": f"{self.model_dir}/topic-docs.txt",
            "diagnostics-file": f"{self.model_dir}/diagnostics.xml",
        }
        for k, v in kwargs.items():
            k = k.replace("_", "-")
            if k in [
                "diagnostics-file",
                "input",
                "output-doc-topics",
                "output-state",
                "output-topic-docs",
                "output-topic-keys",
                "word-topic-counts-file",
            ]:
                v = ensure_posix(v)
            if k is not None:
                opts[k] = v
        return opts

    def _track_progress(self, mallet_cmd: str, num_iterations: int) -> None:
        """Track the progress of the modeling.

        Args:
            mallet_cmd (str): The MALLET command to run.
            num_iterations (int): The number of iterations to run.
        """
        p = Popen(mallet_cmd, stdout=PIPE, stderr=STDOUT, shell=True)
        ll = []
        prog = re.compile(r"\<([^\)]+)\>")
        while p.poll() is None:
            line = p.stdout.readline().decode()
            print(line, end="")
            # Keep track of LL/topic.
            try:
                this_ll = float(re.findall(r"([-+]\d+\.\d+)", line)[0])
                ll.append(this_ll)
            except IndexError:  # Not every line will match.
                pass
            # Keep track of modelling progress
            try:
                this_iter = float(prog.match(line).groups()[0])
                progress = int(100.0 * this_iter / num_iterations)
                if progress % 10 == 0:
                    print("Modeling progress: {0}%.\r".format(progress), end="")
            except AttributeError:  # Not every line will match.
                pass

    @validate_call(config=model_config)
    def import_dir(
        self,
        input: str,
        output: Optional[str] = None,
        keep_sequence: Optional[bool] = False,
        preserve_case: Optional[bool] = False,
        token_regex: Optional[str] = "\\p{L}[\\p{L}\\p{P}]+\\p{L}",
    ) -> None:
        """Import all text files in a directory.

        Args:
            input (str): The path to the directory containing text files.
            output (Optional[str]): The path to the MALLET import file.
            keep_sequence (Optional[bool]): Whether to keep the sequence of the data.
            preserve_case (Optional[bool]): Whether to preserve the case of the data.
            token_regex (Optional[str]): The regex pattern MALLET will use for tokenization.
        """
        # Validate input path
        input = ensure_posix(input)
        if not Path(input).exists():
            raise LexosException(f"Directory not found: {input}.")

        # Get the paths of all the text files in the input directory
        file_paths = glob.glob(input + "/*.txt")
        if len(file_paths) == 0:
            raise LexosException("Directory is empty.")

        # Remove any existing data file
        if Path(self.data_file).exists():
            Path(self.data_file).unlink()

        # Read the files and and save them to the data file
        self.msg.text(f"Importing data from {input}...")
        with open(self.data_file, "a") as data_file:
            for file in file_paths:
                with open(file, "r") as f:
                    text = f.read()
                data_file.write(f"{text}\n")

        # Remove any existing output file
        output = ensure_posix(output)
        if Path(output).exists():
            Path(output).unlink()

        # Import the data
        self._import_data(
            self.data_file, output, keep_sequence, preserve_case, token_regex
        )

    @validate_call(config=model_config)
    def import_docs(
        self,
        docs: list[Doc | Span | str],
        output: str,
        keep_sequence: Optional[bool] = False,
        preserve_case: Optional[bool] = False,
        token_regex: Optional[str] = "\\p{L}[\\p{L}\\p{P}]+\\p{L}",
        use_lemmas: Optional[bool] = False,
        remove_stops: Optional[bool] = False,
        remove_punct: Optional[bool] = False,
        allowed: Optional[list[str]] = None,
    ) -> None:
        """Import a list of docs or strings.

        Args:
            input (str): The path to the directory containing text files.
            output (str): The path to the MALLET import file.
            keep_sequence (Optional[bool]): Whether to keep the sequence of the data.
            preserve_case (Optional[bool]): Whether to preserve the case of the data.
            token_regex (Optional[str]): The regex pattern MALLET will use for tokenization.
            use_lemmas (Optional[bool]): Whether to use lemmas.
            remove_stops (Optional[bool]): Whether to remove stop words.
            remove_punct (Optional[bool]): Whether to remove punctuation.
            allowed (Optional[list[str]]): A list of allowed POS tags.
        """
        # Remove any existing data file
        if Path(self.data_file).exists():
            Path(self.data_file).unlink()

        with open(self.data_file, "a") as data_file:
            for doc in docs:
                if isinstance(doc, str):
                    data_file.write(f"{doc.strip()}\n")
                else:
                    if allowed:
                        Token.set_extension(
                            "is_allowed",
                            getter=lambda token: token.pos_ in allowed,
                            force=True,
                        )
                    else:
                        Token.set_extension("is_allowed", default=True, force=True)
                    bag = self._get_token_bag(
                        doc, use_lemmas, remove_stops, remove_punct
                    )
                    data_file.write(f"{bag.strip()}\n")

        # Remove any existing output file
        output = ensure_posix(output)
        if Path(output).exists():
            Path(output).unlink()

        # Import the data
        self._import_data(
            self.data_file, output, keep_sequence, preserve_case, token_regex
        )

    def import_file(
        self,
        input: str,
        output: str,
        keep_sequence: bool,
        preserve_case: bool,
        token_regex: str = r"\p{L}[\p{L}\p{P}]+\p{L}",
    ) -> None:
        """Import data from an existing data file.

        Args:
            input (str): The path to the directory containing text files.
            output (str): The path to the MALLET import file.
            keep_sequence (Optional[bool]): Whether to keep the sequence of the data.
            preserve_case (Optional[bool]): Whether to preserve the case of the data.
            token_regex (Optional[str]): The regex pattern MALLET will use for tokenization.
        """
        # Ensure the input out and output paths are in posix format
        input = ensure_posix(input)
        output = ensure_posix(output)

        # Import the data
        if Path(input).exists():
            self._import_data(input, output, keep_sequence, preserve_case, token_regex)
        else:
            self.msg.fail(f"File not found: {input}.")

    @validate_call(config=model_config)
    def train(
        self,
        mallet_import_file: Optional[str] = None,
        num_topics: Optional[int] = 20,
        num_iterations: Optional[int] = 1000,
        optimize_interval: Optional[int] = 10,
        random_seed: Optional[int] = None,
        verbose: Optional[bool] = True,
        **kwargs,
    ) -> None:
        """Train the model.

        Args:
            mallet_import_file (Optional[str]): The path to the MALLET import file.
            num_topics (Optional[int]): The number of topics to model.
            num_iterations (Optional[int]): The number of iterations to run.
            optimize_interval (Optional[int]): The interval to optimize the model.
            random_seed (Optional[int]): The random seed for the model.
            verbose (Optional[bool]): Whether to show verbose output.
            **kwargs: Additional MALLET `train-topics` options for the model.
        """
        # Set import file
        if mallet_import_file is None and self.mallet_import_file is None:
            self.msg.fail("Please provide a `mallet_import_file`.")
        elif mallet_import_file is None:
            mallet_import_file = self.mallet_import_file
        else:
            mallet_import_file = ensure_posix(mallet_import_file)

        # Set the training options
        opts = self._set_training_options(
            mallet_import_file,
            num_topics,
            num_iterations,
            optimize_interval,
            random_seed,
            **kwargs,
        )

        # Build mallet command
        mallet_cmd = f"{self.mallet_bin} train-topics --input {mallet_import_file}"
        mallet_cmd += " ".join(opts)

        # Train the topics
        self.msg.text(f"Running {mallet_cmd}\n")
        mallet_cmd = shlex.split(mallet_cmd)
        try:
            if verbose:
                self._track_progress(mallet_cmd, num_iterations)
            else:
                _ = check_output(
                    mallet_cmd, stderr=STDOUT, shell=True, universal_newlines=True
                )
            self.msg.good("Model trained successfully.")
        except CalledProcessError as e:
            self.msg.fail = e.output

    @validate_call(config=model_config)
    def scale(self, model_state_file: str = None, output: str = None):
        """Scale a model.

        Args:
            model_state_file (str): The path to a state_file.
            output (str): The path to an output file.
        """
        self.msg.text("Processing...")
        if not output:
            output = f"{self.model_dir}/topic_scaled.csv"
        try:
            # Convert the mallet output_state file to a pyLDAvis data object
            converted_data = scale_model.convert_mallet_data(model_state_file)

            # Get the topic coordinates in a dataframe
            topic_coordinates = scale_model.get_topic_coordinates(**converted_data)

            # Save the topic coordinates to a CSV file
            topic_coordinates.to_csv(output, index=False, header=False)
            self.msg.good("Done!")
        except Exception as e:
            raise LexosException(f"Failed!: {e}")
