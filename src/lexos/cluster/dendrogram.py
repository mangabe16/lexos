"""dendrogram.py.

Last Updated: February 17, 2025
Last Tested: TBD

# Some sample data for testing
import spacy
from lexos.dtm import DTM
nlp = spacy.load("en_core_web_sm")

dtm = DTM()
dtm(
    docs=[nlp("kitten alert"), nlp("term1"), nlp("Term3"), nlp("10term"), nlp("2term")],
    labels=["doc1", "doc2", "doc3", "doc4", "doc5"],
)
"""

from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, validate_call
from scipy.spatial.distance import pdist

from lexos.dtm import DTM
from lexos.exceptions import LexosException


class Dendrogram(BaseModel):
    """Dendrogram.

    Typical usage:

    ```python
    from lexos.cluster.dendrogram import Dendrogram

    dendrogram = Dendrogram(dtm, show=True)

    or

    dendrogram = Dendrogram(dtm, show=False)
    dendrogram.fig
    ```
    """

    dtm: Optional[ArrayLike | pd.DataFrame] = Field(
        None, json_schema_extra={"The document-term matrix."}
    )
    labels: Optional[list[str]] = Field(
        None, json_schema_extra={"description": "The labels for the dendrogram."}
    )
    metric: Optional[str] = Field(
        "euclidean",
        json_schema_extra={"description": "The metric to use for the dendrogram."},
    )
    method: Optional[str] = Field(
        "average",
        json_schema_extra={"description": "The method to use for the dendrogram."},
    )
    truncate_mode: Optional[str] = Field(
        None, json_schema_extra={"description": "The truncate mode for the dendrogram."}
    )
    color_threshold: Optional[str] = Field(
        None,
        json_schema_extra={"description": "The color threshold for the dendrogram."},
    )
    get_leaves: Optional[bool] = Field(
        True, json_schema_extra={"description": "The get leaves for the dendrogram."}
    )
    orientation: Optional[str] = Field(
        "top", json_schema_extra={"description": "The orientation for the dendrogram."}
    )
    count_sort: Optional[bool | str] = Field(
        None, json_schema_extra={"description": "The count sort for the dendrogram."}
    )
    distance_sort: Optional[bool | str] = Field(
        None, json_schema_extra={"description": "The distance sort for the dendrogram."}
    )
    show_leaf_counts: Optional[bool] = Field(
        False,
        json_schema_extra={"description": "The show leaf counts for the dendrogram."},
    )
    no_plot: Optional[bool] = Field(
        False, json_schema_extra={"description": "The no plot for the dendrogram."}
    )
    no_labels: Optional[bool] = Field(
        False, json_schema_extra={"description": "The no labels for the dendrogram."}
    )
    leaf_rotation: Optional[int] = Field(
        90, json_schema_extra={"description": "The leaf rotation for the dendrogram."}
    )
    leaf_font_size: Optional[int] = Field(
        None,
        json_schema_extra={"description": "The leaf font size for the dendrogram."},
    )
    leaf_label_func: Optional[Callable] = Field(
        None,
        json_schema_extra={
            "description": "The leaf label function for the dendrogram."
        },
    )
    show_contracted: Optional[bool] = Field(
        False,
        json_schema_extra={"description": "The show contracted for the dendrogram."},
    )
    link_color_func: Optional[Callable] = Field(
        None,
        json_schema_extra={
            "description": "The link color function for the dendrogram."
        },
    )
    ax: Optional[Axes] = Field(
        None, json_schema_extra={"description": "The ax for the dendrogram."}
    )
    above_threshold_color: Optional[str] = Field(
        "C0",
        json_schema_extra={
            "description": "The above threshold color for the dendrogram."
        },
    )
    title: Optional[str] = Field(
        None, json_schema_extra={"description": "The title for the dendrogram."}
    )
    figsize: Optional[tuple] = Field(
        (10, 10), json_schema_extra={"description": "The figsize for the dendrogram."}
    )
    show: Optional[bool] = Field(
        False, json_schema_extra={"description": "The show for the dendrogram."}
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self) -> None:
        """Initialise the Dendrogram."""
        pass
        # Create an empty plot for matplotlib
        # Get the dtm table of counts with documents as columns and terms as rows (and transpose)
        # Replace with dtm.to_df(transpose=True)
        # Allow the user to pass a list of dicts as well. How does scipy handle this?
        # self.dtm_table = dtm.get_table()
        # Set "terms" as the index and transpose the table
        # self.dtm_table = self.dtm_table.set_index("terms").T

    def __call__(self, dtm: ArrayLike | pd.DataFrame, **kwargs):
        """Build a dendrogram."""
        # Create the distance and linkage matrixes for matplotlib
        # Requires an m by n array of m original observations in an n-dimensional space.
        # an array-like is any Python object that np.array can convert to an ndarray.
        # This includes nested lists, tuples, scalars and existing arrays.
        if not dtm:
            dtm = self.dtm
        for key, value in kwargs.items():
            setattr(self, key, value)
        # WARNING: I don't like this. You should be able to pass:
        # - a DTM object
        # - a list of lists
        # - a numpy array
        # - a pandas dataframe
        if isinstance(dtm, pd.DataFrame):
            matrix = self._get_matrix_from_dtm(dtm)
        X = pdist(matrix, metric=self.metric)
        Z = sch.linkage(X, self.method)
        fig, ax = plt.subplots(figsize=self.figsize)
        if self.title:
            plt.title(self.title)
        sch.dendrogram(
            Z,
            labels=self._get_column_labels(),
            truncate_mode=self.truncate_mode,
            color_threshold=self.color_threshold,
            get_leaves=self.get_leaves,
            orientation=self.orientation,
            count_sort=self.count_sort,
            distance_sort=self.distance_sort,
            show_leaf_counts=self.show_leaf_counts,
            no_plot=self.no_plot,
            no_labels=self.no_labels,
            leaf_rotation=self.leaf_rotation,
            leaf_font_size=self.leaf_font_size,
            leaf_label_func=self.leaf_label_func,
            show_contracted=self.show_contracted,
            link_color_func=self.link_color_func,
            ax=self.ax,
            above_threshold_color=self.above_threshold_color,
        )
        self.fig = fig

        if not self.show:
            plt.close()

    def _get_matrix_from_dtm(dtm: DTM) -> pd.DataFrame:
        df = dtm.to_df()
        df.index.name = "terms"
        return df.T

    def _get_column_labels(self) -> list[str]:
        """Use default labels from the DTM table.

        Returns:
            A list of column labels.

        Note: You can get labels from a DTM object with dtm.labels.
        From a dataframe with dtm.columns.values.tolist().
        From a list of lists or numpy array, use labels = [f"Doc{i + 1}" for i in range(len(matrix[0]) - 1)].
        Or have the user supply a list of labels that should be the same length as the number of columns in the matrix.
        """
        if not self.labels:
            try:
                self.labels = self.dtm_table.columns.values.tolist()[1:]
            except LexosException:
                # TODO: Find a way to auto-generate labels as "Doc1", "Doc2", etc.
                # Hard to do if you don't know that your dtm is a dataframe
                if isinstance(self.dtm, pd.DataFrame):
                    self.labels = self.dtm.columns.values.tolist()[1:]
                else:
                    self.labels = [f"Doc{i}" for i, _ in enumerate(self.dtm)]

                self.labels = []
        return self.labels()

    @validate_call
    def savefig(self, path: Path | str):
        """Show the figure if it is hidden.

        Args:
            path (Path | str): The path to the file to save.
        """
        self.fig.savefig(path)

    def showfig(self):
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure
        using `Dendrogram.fig`. This will generally display in a
        Jupyter notebook.
        """
        return self.fig
