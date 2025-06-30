import pandas as pd
from spacy.tokens import Doc
from lexos.corpus.record import Record
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
        # create dataframe from the two lists: filenames and predictions
        df = pd.DataFrame({
            'filename': filenames,
            'prediction': predictions
        })

        # save the dataframe to a CSV file
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
        # Holds the of lexos record objects
        records = []

        # loop through the documents and their corresponding labels and predictions
        for i, (doc, label, pred) in enumerate(zip(docs, labels, predictions)):
            # so that it is stored in spacys standard classification format
            # says the doc belongs to the predicted category with confidence 1.0
            doc.cats = {pred: 1.0}

            # add custom label for the prediction inside the spacy docs user data
            # This is just a plain string for easy access later
            doc.user_data["classification_label"] = pred

            #create a metadata dictionary with the predicted class
            #this will be stored in the Lexos Record object
            meta = {"predicted_class": pred}

            # wrap the document and its metadata in a lexos record object
            # The Record class expects a name, content, and meta information
            # The content is the spaCy Doc object, which can be processed later
            # The name is the label (filename or identifier) for the document
            # The meta dictionary contains the predicted class label
            record = Record(name=label, content=doc, meta={"classification": meta})
            records.append(record)

        print(f"{len(records)} records created with classification metadata.")
        return records
