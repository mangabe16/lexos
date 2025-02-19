"""__init__.py.

Last Update: February 17, 2025
Last Tested: February 18, 2025
"""

from typing import Callable, Iterable, Literal, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from natsort import natsorted, ns
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc
from textacy.representations.vectorizers import Vectorizer as TextacyVectorizer

from lexos.constants import SORTING_ALGORITHM
from lexos.exceptions import LexosException

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema
)


class Vectorizer(BaseModel):
    """Wrapper class for Textacy's Vectorizer."""

    @validate_call(config=validation_config)
    def __call__(
        self,
        *,
        tf_type: Literal["linear", "sqrt", "log", "binary"] = "linear",
        idf_type: Optional[Literal["linear", "sqrt", "log"]] = None,
        dl_type: Optional[Literal["linear", "sqrt", "log"]] = None,
        norm: Optional[Literal["l1", "l2"]] = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_n_terms: Optional[int] = None,
        vocabulary_terms: Optional[dict[str, int] | Iterable[str]] = None,
    ) -> TextacyVectorizer:
        """Return a Textacy Vectorizer object.

        Args:
            tf_type (str): Term frequency type.
            idf_type (str): Inverse document frequency type.
            dl_type (str): Document length type.
            norm (str): Normalization type.
            min_df (int | float): Minimum document frequency.
            max_df (int | float): Maximum document frequency.
            max_n_terms (int): Maximum number of terms.
            vocabulary_terms (dict | Iterable[str]): Vocabulary terms.

        Returns:
            TextacyVectorizer: A Textacy Vectorizer object.
        """
        return TextacyVectorizer(
            tf_type=tf_type,
            idf_type=idf_type,
            dl_type=dl_type,
            norm=norm,
            min_df=min_df,
            max_df=max_df,
            max_n_terms=max_n_terms,
            vocabulary_terms=vocabulary_terms,
        )


class DTM(BaseModel):
    """Class for a document-term matrix."""

    docs: Optional[list[list[str] | Doc]] = Field(
        default=None, json_schema_extra={"description": "A list of spaCy docs or a list of token lists."}
    )
    labels: Optional[list[str]] = Field(
        default=None, json_schema_extra={"description": "A list of labels for the documents."}
    )
    vectorizer: Optional[Callable] = Field(
        default=TextacyVectorizer,
        json_schema_extra={"description": "A callable Vectorizer. Must have a fit_transform() method."}
    )
    alg: Optional[ns] = Field(
        default=ns.LOCALE, json_schema_extra={"description": "The sorting algorithm to use."}
    )
    doc_term_matrix: Optional[sp.spmatrix] = Field(
        default=None, json_schema_extra={"description": "The document-term matrix."}
    )

    model_config = validation_config

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the DTM.

        Returns:
            tuple[int, int]: The shape of the DTM.
        """
        return self.doc_term_matrix.get_shape()

    def __call__(
        self, docs: Optional[list[list[str] | Doc]], labels: Optional[Iterable[str]]
    ) -> None:
        """Call method for DTM class.

        Args:
            docs (list[list[str] | Doc]): A list of spaCy docs or a list of token lists.
            labels (list[str]): A list of labels for the documents.

        Note:
            - If you want to filter the docs by token attributes, you can do so beforehand
            and pass the filtered docs to this method.
            - If you want to sort the dataframe, use pandas sort_values(), but make sure to
              pass `SORTING_ALGORITHM` or `self.alg` to the `key` parameter for natsorting.
        """
        if docs is None or len(docs) == 0:
            raise LexosException(
                "You must provide a list of docs or a list of lists of token strings."
            )

        # Make sure the sorting algorithm is valid
        self._validate_sorting_algorithm()

        # Coerce the docs to a list of token lists
        if docs:
            self.docs = docs
        self.docs = [
            [token.text for token in doc] if isinstance(doc, Doc) else doc
            for doc in self.docs
        ]
        # Set the instance labels
        if labels:
            self.labels = labels
        elif self.labels is None:
            self.labels = [f"Doc{i + 1}" for i in range(len(self.docs))]

        # Make sure the number of docs matches the number of labels
        if len(self.docs) != len(self.labels):
            raise LexosException("The number of docs must match the number of labels.")

        # Call the vectorizer to build the DTM
        try:
            self.vectorizer = self.vectorizer()
            self.doc_term_matrix = self.vectorizer.fit_transform(self.docs)
        except Exception as e:
            raise LexosException(f"Error building DTM: {e}")

    @property
    def sorted_terms_list(self) -> list[str]:
        """Return a natsorted list of terms in the DTM.

        Returns:
            list[str]: A natsorted list of terms in the DTM.
        """
        return natsorted(self.vectorizer.terms_list, reverse=False, alg=self.alg)

    @property
    def sorted_term_counts(self) -> dict[str, int]:
        """Return a natsorted dict of terms and their counts in the DTM.

        Returns:
            dict[str, int]: A natsorted dict of terms and their counts.
        """
        terms_counts = natsorted(
            self.vectorizer.vocabulary_terms.items(), reverse=False, alg=self.alg
        )
        return {term: count for term, count in terms_counts}

    def _get_term_percentages(
        self,
        df: pd.DataFrame,
        rounding: int = 3,
        as_str: bool | str = "string",
        sum: bool = False,
        mean: bool = False,
        median: bool = False,
    ) -> pd.DataFrame:
        """Return a dataframe with term frequencies.

        Args:
            df (pd.DataFrame): The dataframe to convert to percentages.
            rounding (int): The number of decimal places to round to.
            as_str (bool | str): Whether to return the terms as strings.
            sum (bool): Whether to include a column for the sum of each row.
            mean (bool): Whether to include a column for the mean of each row.
            median (bool): Whether to include a column for the median of each row.

        Returns:
            pd.DataFrame: A dataframe with term frequencies.
        """
        dense_array = df.to_numpy()
        total_sum = np.sum(dense_array)
        if total_sum != 0:
            percentage_array = (dense_array / total_sum) * 100
        else:
            percentage_array = np.zeros_like(dense_array)
        df = pd.DataFrame(percentage_array, columns=df.columns, index=df.index)
        if sum:
            df["Total"] = df.sum(numeric_only=True, axis=1)
        if mean:
            df["Mean"] = df.mean(numeric_only=True, axis=1)
        if median:
            df["Median"] = df.median(numeric_only=True, axis=1)
        if rounding:
            df = df.round(rounding)
        if as_str == "string":
            df = df.map(lambda x: f"{x}%")
        return df

    def _validate_sorting_algorithm(self) -> bool:
        """Ensure that the specified sorting algorithm is a valid natsort locale.

        Args:
            alg: The sorting algorithm to validate.

        Returns:
            bool: Whether the sorting algorithm is valid.
        """
        if self.alg not in [e for e in ns]:
            locales = ", ".join([f"ns.{e.name}" for e in ns])
            err = (
                f"Invalid sorting algorithm: {self.alg}.",
                f"Valid algorithms for `alg` are: {locales}.",
                "See https://natsort.readthedocs.io/en/stable/api.html#natsort.ns.",
            )
            raise LexosException(" ".join(err))
        return True

    def to_df(
        self,
        by: Optional[list | list[str]] = None,
        ascending: Optional[bool | list[bool]] = True,
        as_percent: Optional[bool] = False,
        rounding: Optional[int] = 3,
        transpose: Optional[bool] = False,
        sum: Optional[bool] = False,
        mean: Optional[bool] = False,
        median: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Return the whole DTM as a pandas dataframe.

        Args:
            by (Optional[list | list[str]]): The column(s) to sort by.
            ascending (Optional[bool | list[bool]]): Whether to sort in ascending order.
            as_percent (Optional[bool]): Whether to return the terms as percentages.
            rounding (Optional[int]): The number of decimal places to round to.
            transpose (Optional[bool]): Whether to transpose the dataframe.
            sum (Optional[bool]): Whether to include a column for the sum of each row.
            mean (Optional[bool]): Whether to include a column for the mean of each row.
            median (Optional[bool]): Whether to include a column for the median of each row.

        Returns:
            pd.DataFrame: The DTM as a pandas dataframe.
        """
        if by is None:
            by = self.labels[0]
        df = pd.DataFrame.sparse.from_spmatrix(
            self.doc_term_matrix, columns=self.vectorizer.terms_list, index=self.labels
        ).T
        df = df.sort_values(by=by, ascending=ascending)
        if as_percent:
            df = self._get_term_percentages(
                df,
                rounding=rounding,
                as_str=as_percent,
                sum=sum,
                mean=mean,
                median=median,
            )
        else:
            if sum:
                df["Total"] = df.sum(numeric_only=True, axis=1)
            if mean:
                df["Mean"] = df.mean(numeric_only=True, axis=1)
            if median:
                df["Median"] = np.median(df.to_numpy())
        if transpose:
            df = df.T
        return df
