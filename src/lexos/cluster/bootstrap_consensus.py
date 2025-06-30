"""This is a model to produce bootstrap consensus tree of the dtm.

Last update: March 6, 2025
Last tested: March 6, 2025

See https://github.com/koonimaru/omniplot/blob/962310436a153098b671ebd76cdd59f8a7b5e681/omniplot/plot.py#L38 for a method of getting round dendrograms.
"""

from io import StringIO
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.Consensus import majority_consensus
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, field_validator, validate_call
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.util import is_valid_colour

PRECISION = 1  # Precision for branch length formatting in dendrogram labels


class BCT(BaseModel):
    """The Bootstrap Consensus Tree Class."""

    doc_term_matrix: Optional[DTM] = Field(
        None, json_schema_extra={"description": "The document term matrix."}
    )
    distance_metric: Optional[str] = Field(
        "euclidean", json_schema_extra={"description": "The distance metric."}
    )
    linkage_method: Optional[str] = Field(
        "average", json_schema_extra={"description": "The linkage method."}
    )
    cutoff: Optional[float] = Field(
        0.5, json_schema_extra={"description": "The cutoff value."}
    )
    iterations: Optional[int] = Field(
        100,
        json_schema_extra={
            "description": "The number of iterations to run the bootstrap."
        },
    )
    replace: Optional[str] = Field(
        "without", json_schema_extra={"description": "The replacement method."}
    )
    doc_labels: Optional[list[int | str] | dict[int, str]] = Field(
        None, json_schema_extra={"description": "The document labels."}
    )
    text_color: Optional[str] = Field(
        "rgb(0, 0, 0)", json_schema_extra={"description": "The text colour."}
    )
    showfig: Optional[bool] = Field(
        False, json_schema_extra={"description": "Whether to show the figure."}
    )
    fig: Optional[Figure] = Field(
        None, json_schema_extra={"description": "The figure for the dendrogram."}
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("text_color", mode="after")
    @classmethod
    def _validate_text_color(cls, value):
        if not is_valid_colour(value):
            raise LexosException(
                "Value is not a valid colour: string not recognised as a valid colour."
            )
        return value

    @property
    def _doc_term_matrix(self) -> pd.DataFrame:
        """Return a dataframe of the document term matrix.

        Returns:
            pd.DataFrame: The document term matrix with doc labels as the index and terms as the columns.


        Note that the web app uses doc ids as the index.
        """
        if self.doc_term_matrix is None:
            raise LexosException("No document term matrix found.")
        return self.doc_term_matrix.to_df().T

    @property
    def _document_label_map(self) -> dict[int, str] | dict:
        """Return a dictionary of document label map.

        Returns:
            list[int | str] | dict[int, str]: A document label map or a list of indices or labels.
        """
        if self.doc_labels is not None and len(self.doc_labels) > 0:
            if isinstance(self.doc_labels, dict):
                return self.doc_labels
            else:
                if isinstance(self.doc_labels[0], int):
                    return {i: f"doc{i + 1}" for i, _ in enumerate(self.doc_labels)}
                else:
                    return {i: label for i, label in enumerate(self.doc_labels)}
        return {}

    @staticmethod
    def linkage_to_newick(matrix: np.ndarray, labels: list[str]) -> str:
        """Convert a linkage matrix to a Newick formatted tree.

        Args:
            matrix (np.ndarray): The linkage matrix.
            labels (list[str]): Names of the tree node.

        Returns:
            str: The Newick representation of the linkage matrix.
        """
        # Convert the linkage matrix to a ClusterNode object
        tree = to_tree(matrix, False)

        # Define a recursive function to build the Newick tree
        def _build_newick_tree(
            node: ClusterNode, newick: str, parent_dist: float, leaf_names: list[str]
        ) -> str:
            """Recursively build the Newick tree.

            Args:
                node (ClusterNode): The tree node currently being converted to Newick.
                newick (str): The current Newick representation of the tree.
                parent_dist (float): The distance to parent node.
                leaf_names (list[str]): Names of the tree node.

            Returns:
                str: The Newick representation of the tree.
            """
            # If the node is a leaf, enclose
            if node.is_leaf():
                return f"{leaf_names[node.id]}:{(parent_dist - node.dist) / 2}{newick}"
            else:
                # Write the distance to the parent node
                newick = (
                    f"):{(parent_dist - node.dist) / 2}{newick}"
                    if len(newick) > 0
                    else ");"
                )
                # Recursive call to expand the tree
                newick = _build_newick_tree(
                    newick=newick,
                    node=node.get_left(),
                    parent_dist=node.dist,
                    leaf_names=leaf_names,
                )
                newick = _build_newick_tree(
                    newick=f",{newick}",
                    node=node.get_right(),
                    parent_dist=node.dist,
                    leaf_names=leaf_names,
                )
                # Enclose the tree at the beginning
                return f"({newick}"

        # Trigger the recursive function
        return _build_newick_tree(
            node=tree, newick="", parent_dist=tree.dist, leaf_names=labels
        )

    def _get_newick_tree(self, labels: list[str], sample_dtm: pd.DataFrame) -> str:
        """Get Newick tree based on a subset of the DTM.

        Args:
            labels (list[str]): All file names from the DTM
            sample_dtm (pd.DataFrame): An 80% subset of the complete DTM

        Returns:
            str: A Newick formatted tree representing the DTM subset
        """
        # Get the linkage matrix for the sample doc term matrix
        linkage_matrix = linkage(
            sample_dtm.values, metric=self.distance_metric, method=self.linkage_method
        )

        # Get the Newick representation of the tree
        newick = self.linkage_to_newick(matrix=linkage_matrix, labels=labels)

        # Convert linkage matrix to a tree node and return it
        return Phylo.read(StringIO(newick), format="newick")

    def _get_bootstrap_trees(self) -> list[str]:
        """Do bootstrap on the DTM to get a list of Newick trees.

        Returns:
            list[str]: A list of Newick formatted tree where each tree was based on an 80% subset of the complete DTM.
        """
        # Save the DTM to avoid multiple calls
        doc_term_matrix = self._doc_term_matrix

        # Get doc names, since tree nodes need labels
        labels = [doc for doc in self._doc_term_matrix.index.values.tolist()]

        # The bootstrap process to get all the trees.
        return [
            self._get_newick_tree(
                sample_dtm=doc_term_matrix.sample(
                    axis=1,
                    frac=0.8,
                    replace=self.replace,
                    random_state=np.random.RandomState(),
                ),
                labels=labels,
            )
            for _ in range(self.iterations)
        ]

    def _get_bootstrap_consensus_tree(self) -> Phylo:
        """Get the consensus tree.

        Returns:
            Phylo: The consensus tree of the list of Newick trees.
        """
        # Find the consensus of all the Newick trees
        return majority_consensus(trees=self._get_bootstrap_trees(), cutoff=self.cutoff)

    def _get_bootstrap_consensus_tree_fig(self) -> Figure:
        # Get the colours
        color = tuple(map(int, self.text_color[4:-1].split(",")))
        normalized_color = tuple(x / 255 for x in color)

        # Draw the consensus tree as a MatPlotLib object
        tree = self._get_bootstrap_consensus_tree()
        tree.root.color = color

        fig, ax = plt.subplots()

        Phylo.draw(
            tree,
            axes=ax,
            do_show=False,
            branch_labels=lambda clade: "{0:.{PRECISION}f}\n".format(
                clade.branch_length
            )
            if clade.branch_length is not None
            else "",
        )

        # Set labels for the plot
        plt.xlabel("Branch Length", color=normalized_color)
        plt.ylabel("Documents", color=normalized_color)

        # Hide the two unused borders
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # Set the colour of the used borders and labels
        plt.gca().spines["bottom"].set_color(normalized_color)
        plt.gca().spines["left"].set_color(normalized_color)
        plt.gca().tick_params(colors=normalized_color)

        # Extend the x-axis to the right to fit longer labels
        x_left, x_right, y_low, y_high = plt.axis()
        plt.axis((x_left, x_right * 1.25, y_low, y_high))

        # Set the graph size, title, and tight layout
        plt.gcf().set_size_inches(w=9.5, h=(len(self._document_label_map) * 0.3 + 1))
        plt.title("Bootstrap Consensus Tree Result", color=normalized_color)
        plt.gcf().tight_layout()

        # Change the line spacing
        for text in plt.gca().texts:
            text.set_linespacing(spacing=0.1)
            text.set_color(normalized_color)

        if not self.showfig:
            plt.close()

        return fig

    @validate_call(config=model_config)
    def __call__(
        self,
        doc_term_matrix: Optional[DTM] = Field(
            None, json_schema_extra={"description": "The document term matrix."}
        ),
        distance_metric: Optional[str] = Field(
            "euclidean", json_schema_extra={"description": "The distance metric."}
        ),
        linkage_method: Optional[str] = Field(
            "average", json_schema_extra={"description": "The linkage method."}
        ),
        cutoff: Optional[float] = Field(
            0.5, json_schema_extra={"description": "The cutoff value."}
        ),
        iterations: Optional[int] = Field(
            100,
            json_schema_extra={
                "description": "The number of iterations to run the bootstrap."
            },
        ),
        replace: Optional[str] = Field(
            "without", json_schema_extra={"description": "The replacement method."}
        ),
        doc_labels: Optional[list[int | str] | dict[int, str]] = Field(
            None, json_schema_extra={"description": "The document labels."}
        ),
        text_color: Optional[str] = Field(
            "rgb(0, 0, 0)", json_schema_extra={"description": "The text colour."}
        ),
        showfig: Optional[bool] = Field(
            False, json_schema_extra={"description": "Whether to show the figure."}
        ),
    ) -> Figure:
        """Render the bootstrap consensus tree result and save it to images.

        Returns:
            Figure: The rendered BCT figure.
        """
        # Set the attributes of the class
        self._set_attrs(
            doc_term_matrix=doc_term_matrix,
            distance_metric=distance_metric,
            linkage_method=linkage_method,
            cutoff=cutoff,
            iterations=iterations,
            replace=replace,
            doc_labels=doc_labels,
            text_color=text_color,
            showfig=showfig,
        )

        # Do not automatically show the figure
        if not self.showfig:
            plt.ioff()

        # Get the matplotlib figure for bootstrap consensus tree result
        # self._get_bootstrap_consensus_tree_fig() <-- comment this out for now, seems to show 2 figs
        fig = self._get_bootstrap_consensus_tree_fig()

        # Save the figure to the instance and show or close the plot
        self.fig = fig

        return fig

    def _set_attrs(self, **kwargs):
        """Set the attributes of the class."""
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    @validate_call(config=model_config)
    def save(self, path: Path | str) -> None:
        """Save the bootstrap consensus tree result to a file.

        Args:
            path (Path | str): The path to save the file.
        """
        if not path or path == "":
            raise LexosException("You must provide a valid path.")
        self.fig.savefig(path)
        # NOTE: It may be better to avoid plt.ion/ioff by saving a binary
        # version of the image, but that might complicate the ui.
        # Create a bytes IO image holder and save figure to it
        # image_holder = BytesIO()
        # bct_plot.savefig(image_holder, transparent=True)
        # image_holder.seek(0)
        # Decode image to utf-8 string
        # return base64.b64encode(b"".join(image_holder)).decode("utf-8")

    def show(self) -> Figure:
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure using `BCT.fig`.
        This will generally display in a Jupyter notebook.
        """
        if self.fig is None:
            raise LexosException(
                "You must call the instance before showing the figure."
            )
        plt.ion()
        return self.fig
