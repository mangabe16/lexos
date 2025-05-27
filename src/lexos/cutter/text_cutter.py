"""text_cutter.py.

The class has five public methods: `merge()`, `save()`, `split()`, `split_on_milestones()`, and `to_dict()`.
Both methods `split()` and `split_on_milestones()` accept a single source or a list of
sources. By default, sources are assumed to be strings. To accept files, set `file` to
True.

Before using the `save()` method, it is advisable to set the `names` parameter to a list of
names for the sources to be used by the `_write_chunk()` method for determining filenames.

The `merge()` method merges a list of chunks into a single string. TextCutter.merge(chunks)
is equivalent to " ".join(chunks). It exists largely for consistency with the TokenCutter API.

The `split()` by `chunksize` by default. To specify the number of segments, set
`n` to an integer. To split by lines, set `newline` to True. To merge the last
two chunks, set `merge_final` to True.

The `split_on_milestones()` method accepts a list of StringSpan objects as `milestones`.
Be default, milestones are removed from the output, but they they can be retained at
the end of the preceding or the beginning of the following chunk by setting `keep_spans`.

To change the delimiter and padding for the chunk names, set `delimiter` and `pad`.
By default, the `split()` method assumes strings as input. To split files, set `file`
to True. To save the chunks, set `output_dir` to the output directory.

Chunks are stored in the `chunks` attribute, which is a list of lists: each item
in the top-level list is a source; each sub-list contains the chunks of that source.
The class is iterable, so the list can also be accessed with `for chunk in TextCutter`.

The `to_dict()` method returns the chunks as a dictionary with source names as keys.

Last updated: 2025-05-26
Tested: 2025-05-26
"""

import os
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Generator, Iterator, Optional

from pydantic import BaseModel, Field, validate_call

from lexos.constants import LEXOS_MILESTONE_FLAG
from lexos.exceptions import LexosException
from lexos.milestones.string_milestones import StringSpan
from lexos.util import ensure_list

class TextCutter(BaseModel, validate_assignment=True):
    """TextCutter class for chunking files and strings containing untokenised text."""

    chunks: list[list[str]] = []

    chunksize: Optional[int] = Field(
        default=1_000_000,
        json_schema_extra={"description": "The desired chunk size in bytes."},
    )
    n: Optional[int] = Field(
        default=None, json_schema_extra={"description": "The number of chunks."}
    )
    names: Optional[list[str | None]] = Field(
        default=[],
        json_schema_extra={
            "description": "A list of names for the source files/strings."
        },
    )
    newline: Optional[bool] = Field(
        default=False, json_schema_extra={"description": "Whether to chunk by lines."}
    )
    output_dir: Optional[Path | str] = Field(
        default=None,
        json_schema_extra={
            "description": "The output directory to save the chunks to."
        },
    )
    merge_threshold: Optional[float] = Field(
        default=0.5,
        json_schema_extra={
            "description": "The threshold for merging the last two chunks."
        },
    )
    merge_final: Optional[bool] = Field(
        default=False,
        json_schema_extra={"description": "Whether to merge the last two chunks."},
    )
    delimiter: str = Field(
        default="_",
        json_schema_extra={"description": "The delimiter to use for the chunk names."},
    )
    pad: int = Field(
        default=3, json_schema_extra={"description": "The padding for the chunk names."}
    )
    strip_chunks: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Whether to strip leading and trailing whitespace in the chunks."
        },
    )

    def __iter__(self) -> Iterator:
        """Make the class iterable.

        Returns:
            Iterator: An iterator containing the object's chunks.
        """
        return iter([chunk for chunk in self.chunks])

    def __len__(self):
        """Return the number of sources in the instance."""
        return len(self.chunks)

    def _calculate_chunk_size(self, size: int, n: int) -> tuple[int, int]:
        """Calculate chunk size and remainder for n chunks.

        Args:
            size (int): Total size of file in bytes.
            n (int): Number of chunks to create.

        Returns:
            tuple [int, int]: (chunk_size, remainder)
        """
        chunk_size = size // n
        remainder = size % n
        return chunk_size, remainder

    def _get_name(self, source: Path | str, index: int) -> str:
        """Generate a filename based on source or fallback rules.

        Args:
            source (Path | str): Original file path or source label.
            index (int): Index of the source being processed.

        Returns:
            str: A formatted name for saving the chunked output.
        """
        if self.names:
            return self.names[index]
        elif isinstance(source, Path):
            return Path(source).stem
        else:
            return f"text{str(index).zfill(self.pad)}"

    def _merge_final_chunks(
        self, chunks: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """Merge the last two chunks if the final one is below the merge threshold.

        Args:
            chunks (Generator[str]): Chunks of text to evaluate.

        Yields:
            str: Finalized chunks after merging (if needed).
        """
        buffer = []
        for item in chunks:
            buffer.append(item)
            if len(buffer) > 2:
                yield buffer.pop(0)
        if len(buffer) == 2:
            yield "".join([buffer[0], buffer[1]])
        elif len(buffer) == 1:
            yield buffer[0]

    def _process_buffer(
        self,
        source: bytes | str,
        n: bool = False,
    ) -> list[str]:
        """Process single buffer in chunks.

        Args:
            source (bytes | str): The string or bytes source.
            n (bool): Whether to chunk by n.

        Returns:
            list[str]: The chunks.
        """
        if isinstance(source, str):
            source = source.encode()
        chunks = []
        with BytesIO(source) as buffer:
            if n is True:
                file_size = buffer.getbuffer().nbytes
                chunk_size, remainder = self._calculate_chunk_size(file_size, self.n)
                try:
                    for i in range(self.n):
                        if self.newline:
                            chunk = self._read_by_lines(buffer, chunk_size)
                        else:
                            size = (
                                chunk_size + remainder
                                if i == self.n - 1
                                else chunk_size
                            )
                            chunk = buffer.read(size)
                        if not chunk:
                            break
                        # Convert to string
                        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
                finally:
                    buffer.close()
            else:
                while chunk := (
                    self._read_by_lines if self.newline else self._read_chunks
                )(buffer, self.chunksize):
                    # Convert to string
                    chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
        return chunks

    def _process_file(
        self,
        path: Path | str,
        n: bool = False,
    ) -> list[str]:
        """Split the contents of a file into chunks.

        Args:
            path (Path | str): Path to the input file.
            n (bool): Whether to split into a fixed number of parts.

        Returns:
            list[str]: List of chunked text segments.
        """
        chunks = []
        with open(path, "r") as f:
            if n is True:
                file_size = os.path.getsize(str(path))
                chunk_size, remainder = self._calculate_chunk_size(file_size, self.n)
                try:
                    for i in range(self.n):
                        if self.newline:
                            chunk = self._read_by_lines(f, chunk_size)
                            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
                        else:
                            size = (
                                chunk_size + remainder
                                if i == self.n - 1
                                else chunk_size
                            )
                            chunk = f.read(size)
                            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
                        if not chunk:
                            break
                        chunks.append(chunk)
                finally:
                    f.close()
            else:
                while chunk := (
                    self._read_by_lines if self.newline else self._read_chunks
                )(f, self.chunksize):
                    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
                    chunks.append(chunk)
        return chunks

    def _read_by_lines(self, file_or_buf: BinaryIO, size: int) -> str:
        """Read file by lines up to size limit.

        Args:
            file_or_buf (BinaryIO): The file object or buffer to read from.
            size (int): Maximum bytes to read.

        Returns:
            str: Concatenated lines up to size limit.
        """
        chunks: list[bytes] = []
        bytes_read = 0

        while bytes_read < size and (line := file_or_buf.readline()):
            chunks.append(line.decode("utf-8") if isinstance(line, bytes) else line)
            bytes_read += len(line)

        return "".join(chunks)

    def _read_chunks(self, buffer: BytesIO, size: int) -> bytes:
        """Read a fixed number of bytes from a memory buffer.

        Args:
            buffer (BytesIO): The buffer to read from.
            size (int): Number of bytes to read.

        Returns:
            bytes: A chunk of text from the buffer.
        """
        chunk = buffer.read(size)
        return chunk

    def _set_attributes(self, **data) -> None:
        """Update multiple attributes on the TextCutter instance.
        
        Args:
            **data: Arbitrary keyword arguments matching attribute names.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _write_chunk(
        self, path: Path | str, n: int, chunk: str, output_dir: Path
    ) -> None:
        """Write chunk to file with formatted name.

        Args:
            path (Path | str): The path of the original file.
            n (int): The number of the chunk.
            chunk (str): The chunk to save.
            output_dir (Path): The output directory for the chunk.
        """
        path = Path(path)
        output_file = f"{path.stem}{self.delimiter}{str(n).zfill(self.pad)}.txt"
        output_path = output_dir / output_file
        output_path.write_text(chunk)

    @validate_call
    def merge(self, chunks: list[str], sep: str = " ") -> str:
        """Merge a list of chunks into a single str.

        Args:
            chunks (list[str]): The list of chunks to merge.
            sep (str): The separator to use.

        Returns:
            str: The merged string.
        """
        if len(chunks) == 0:
            raise LexosException("No chunks to merge.")
        return f"{sep}".join(string for string in chunks)

    @validate_call
    def save(
        self,
        output_dir: Path | str,
        names: Optional[list[str]] = None,
        delimiter: Optional[str] = "_",
        pad: Optional[int] = 3,
        strip_chunks: Optional[bool] = True,
    ) -> None:
        """Save the chunks to disk.

        Args:
            output_dir (Path | str): The output directory to save the chunks to.
            names (Optional[list[str]]): The source names.
            delimiter (str): The delimiter to use for the chunk names.
            pad (int): The padding for the chunk names.
            strip_chunks (bool): Whether to strip leading and trailing whitespace in the chunks.
        """
        self._set_attributes(
            output_dir=output_dir,
            delimiter=delimiter,
            names=names,
            pad=pad,
            strip_chunks=strip_chunks,
        )
        if not self.chunks:
            raise LexosException("No chunks to save.")
        if self.names:
            if len(self.names) != len(self.chunks):
                raise LexosException(
                    f"The number of docs in `names` ({len(self.names)}) must equal the number of docs in `chunks` ({len(self.chunks)})."
                )
        else:
            self.names = [
                f"doc{str(i + 1).zfill(self.pad)}" for i in range(len(self.chunks))
            ]
        for i, doc in enumerate(self.chunks):
            for num, chunk in enumerate(doc):
                if strip_chunks:
                    chunk = chunk.strip()
                self._write_chunk(self.names[i], num + 1, chunk, Path(output_dir))

    @validate_call
    def split(
        self,
        sources: Path | str | list[Path | str],
        chunksize: Optional[int] = None,
        names: Optional[str | list[str]] = None,
        n: Optional[int] = None,
        newline: Optional[bool] = None,
        output_dir: Optional[Path | str] = None,
        file: Optional[bool] = False,
        merge_threshold: Optional[float] = 0.5,
        merge_final: Optional[bool] = False,
        delimiter: str = "_",
        pad: int = 3,
    ) -> None:
        """Chunk the file or buffer.

        Args:
            sources (Path | str | list[Path | str]): The file path or buffer.
            chunksize (Optional[int]): The size of the chunks.
            names (Optional[str | list[str | None]]): The source names.
            n (Optional[int]): The number of chunks.
            newline (Optional[bool]): Whether to chunk by lines.
            output_dir (Optional[Path | str]): The output directory to save the chunks to.
            file (Optional[bool]): Whether to chunk the file or buffer.
            merge_threshold (Optional[float]): The threshold for merging the last two chunks.
            merge_final (Optional[bool]): Whether to merge the last two chunks.
            delimiter (str): The delimiter to use for the chunk names.
            pad (int): The padding for the chunk names.
        """
        self._set_attributes(
            names=ensure_list(names),
            n=n,
            newline=newline,
            output_dir=output_dir,
            merge_threshold=merge_threshold,
            merge_final=merge_final,
            delimiter=delimiter,
            pad=pad,
        )
        if chunksize:
            self.chunksize = chunksize
        if names:
            self.names = [Path(name).stem for name in ensure_list(names)]
        for i, source in enumerate(ensure_list(sources)):
            name = self._get_name(source, i)
            split_by_num = False
            if isinstance(self.n, int):
                split_by_num = True
            chunks = (
                self._process_file(source, n=split_by_num)
                if file
                else self._process_buffer(source, n=split_by_num)
            )
            # Calculate the threshold here.
            threshold = self.chunksize * self.merge_threshold
            if self.merge_final is True or len(chunks[-1]) < threshold:
                chunks = list(self._merge_final_chunks(chunks))
            self.names.append(name)
            self.chunks.append(chunks)

    @validate_call
    def split_on_milestones(
        self,
        sources: Path | str | list[Path | str],
        milestones: list[StringSpan],
        names: Optional[Path | str | list[Path | str]] = None,
        keep_spans: Optional[bool | str] = False,
        strip: Optional[bool] = True,
        milestone_flag: Optional[str] = LEXOS_MILESTONE_FLAG,
        output_dir: Optional[Path | str] = None,
        file: Optional[bool] = True,
    ) -> None:
        """Split text at each milestone span, optionally retaining the milestone text.

        Args:
            sources (Path | str | list[Path | str]): List of file paths or buffers.
            milestones (list[StringSpan]): A list of milestone StringSpans to split the text at.
            names (Optional[Path | str | list[Path | str]]): List of source names.
            keep_spans (Optional[bool | str]): Whether to keep the spans in the split strings. Defaults to False.
            strip (Optional[bool]): Whether to strip the text. Defaults to True.
            milestone_flag (Optional[str]): The flag to use for the split. Defaults to LEXOS_MILESTONE_FLAG.
            output_dir (Optional[Path | str]): The output directory to save the chunks to.
            file (Optional[bool]): Set to True if reading from file(s), False for strings.
        """
        if file:
            self.names = [Path(name).stem for name in ensure_list(sources)]
        elif names:
            self.names = [Path(name).stem for name in ensure_list(names)]
        else:
            self.names = [
                f"text{str(i + 1).zfill(self.pad)}" for i in range(len(sources))
            ]
        for i, source in enumerate(ensure_list(sources)):
            text = source
            if file:
                try:
                    with open(source, "r", newline="") as f:
                        text = f.read()
                except BaseException as e:
                    raise LexosException(e)
            if keep_spans == "following":
                for span in milestones:
                    text = text.replace(span.text, f"{milestone_flag}{span.text}")
                chunks = text.split(milestone_flag)
            else:
                chunks = []
                start = 0
                for span in milestones:
                    end = span.start
                    chunks.append(text[start:end])
                    if keep_spans == "preceding":
                        chunks[-1] += text[span.start : span.end + 1]
                    start = span.end + 1
                chunks.append(text[start:])
            if strip:
                chunks = [doc.strip() for doc in chunks]
            self.chunks.append(chunks)
            if output_dir:
                name = self.names[i]
                for num, chunk in enumerate(chunks):
                    self._write_chunk(name, num, chunk, output_dir)

    @validate_call
    def to_dict(
        self, names: Optional[Path | str | list[Path | str]] = None
    ) -> dict[str, list[str]]:
        """Return the chunks as a dictionary.

        Args:
            names (Optional[Path | str | list[Path | str]]): The source names.

        Returns:
            dict[str, list[str]]: The chunks as a dictionary.
        """
        if names:
            self.names = ensure_list(names)
        elif self.names == []:
            self.names = [
                f"text{str(i + 1).zfill(self.pad)}" for i in range(len(self.chunks))
            ]
        return {str(source): chunks for source, chunks in zip(self.names, self.chunks)}