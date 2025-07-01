import pandas as pd
from spacy.tokens import Doc
from lexos.corpus import Record
from typing import List, Optional


class PredictionSaver:
    """
    Utility class for saving classification predictions in different formats.
    """

    @staticmethod
    def save_to_csv(filenames: List[str], predictions: List[str], output_file: str):
        """
        Save filenames and predicted labels to a CSV file.

        Parameters:
            filenames: List of document filenames or identifiers
            predictions: List of predicted labels
            output_file: Path to the output CSV file
        """
        df = pd.DataFrame({
            'filename': filenames,
            'prediction': predictions
        })
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    @staticmethod
    def save_to_records(
        docs: List[Doc],
        labels: List[str],
        predictions: List[str],
        confidences: Optional[List[float]] = None
    ) -> List[Record]:
        """
        Attach classification results to spaCy Docs and return wrapped Lexos Record objects.

        Parameters:
            docs: List of spaCy Doc objects
            labels: Corresponding filenames or document names
            predictions: Predicted class labels
            confidences: Optional confidence scores

        Returns:
            List of Lexos Record objects with classification metadata attached
        """
        records = []

        for i, (doc, label, pred) in enumerate(zip(docs, labels, predictions)):
            # Add classification result to spaCy doc
            doc.cats = {pred: 1.0}
            doc.user_data["classification_label"] = pred

            # Add optional confidence if provided
            meta = {"predicted_class": pred}
            if confidences and i < len(confidences):
                meta["confidence"] = float(confidences[i])

            # Create and store the record
            record = Record(name=label, content=doc, meta={"classification": meta})
            records.append(record)

        print(f"{len(records)} records created with classification metadata.")
        return records
