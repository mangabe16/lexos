"""utils.py.

Last Updated: June 30, 2025
Last Tested: TBD.
"""

import csv
from typing import List, Union, Sequence

import pandas as pd
from spacy.tokens import Doc

# from lexos.corpus import Record


def save_predictions(filenames: list, predictions: list, output_file: str) -> None:
    """Save a list of filenames and their corresponding predicted labels to a CSV file.

    Args:
        filenames (list): list of filenames
        predictions (list): predicted labels for each file
        output_file (str): output CSV file path/name
    """
    # combine filenames and predictions into a pandas DataFrame
    df = pd.DataFrame({"filename": filenames, "prediction": predictions})
    # save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


class PredictionSaver:
    """Simple wrapper class to save predictions (kept for API compatibility)."""

    def __init__(self, default_output: str = "predictions.csv"):
        self.default_output = default_output

    def save(
        self,
        filenames: Sequence[str],
        predictions: Sequence[str],
        output_file: str | None = None,
    ):
        target = output_file or self.default_output
        save_predictions(list(filenames), list(predictions), target)


__all__ = ["save_predictions", "PredictionSaver"]

# def save_predictions(
#     labels: List[str],
#     predictions: List[str],
#     output_path: str,
#     docs: List[Doc] = None,
#     output_format: str = "csv"
# ):
#     """
#     Save predictions to a CSV or attach them to spaCy Docs and return Records.

#     Parameters:
#         labels: List of document names or labels
#         predictions: List of predicted category strings
#         output_path: File path for CSV, ignored for 'record' output
#         docs: List of spaCy Docs (required if output_format is 'record')
#         output_format: 'csv' or 'record'

#     Returns:
#         List of Records if output_format is 'record'; None otherwise
#     """
#     if output_format == "csv":
#         with open(output_path, mode='w', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow(["Label", "Prediction"])
#             writer.writerows(zip(labels, predictions))
#         print(f"Predictions saved to {output_path}")
#         return None

#     elif output_format == "record":
#         if docs is None:
#             raise ValueError("spaCy Docs are required for 'record' output")

#         from lexos.corpus import Record

#         records = []
#         for label, prediction, doc in zip(labels, predictions, docs):
#             doc.cats = {prediction: 1.0}  # mark predicted class
#             doc.user_data["classification_label"] = prediction  # general storage
#             record = Record(name=label, content=doc, meta={"classification": prediction})
#             records.append(record)

#         print(f"{len(records)} Records created with classification metadata.")
#         return records

#     else:
#         raise ValueError(f"Unsupported output_format: {output_format}")
