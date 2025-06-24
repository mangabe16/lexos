from lexos.topwords import TopWords
from spacy.tokens import Doc
from pydantic import Field, ConfigDict
from spacy.schemas import DocJSONSchema
from lexos.tokenizer import Tokenizer
from textacy import extract
import pandas as pd
from typing import Any, Literal
import spacy  # Import spacy
from spacy.lang.en.stop_words import STOP_WORDS  # Import STOP_WORDS

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
)

# register a custom extension for keyterms if not already set
if not Doc.has_extension("keyterms"):  # Changed from "keywords" to "keyterms"
    Doc.set_extension(
        "keyterms", default=None, force=True
    )  # Changed from "keywords" to "keyterms"


class KeyTerms(TopWords):
    """Extracts keyterms from text or a spaCy Doc using textacy algorithms."""

    document: str | Doc | None = Field(
        None, description="The raw text or spaCy doc to analyze."
    )
    method: Literal["textrank", "sgrank"] = Field(
        ...,
        description="Method for keyterm extraction (e.g., 'textrank', 'sgrank').",  # Changed description
    )
    topn: int = Field(
        10, gt=0, description="Number of top keyterms to return."
    )  # Changed description
    model: str = Field(
        default="en_core_web_sm",
        description="spaCy model name to use for tokenization.",
    )
    ngrams: tuple[int, int] = Field(
        default=(1, 3),
        description="The ngram range for keyterm extraction, e.g., (1, 1) for unigrams only.",  # Changed description
    )
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)
    normalize: str | None = Field(
        default=None,
        description="Normalization for keyterm extraction (e.g., 'lemma', 'lower', or None).",
    )

    model_config = validation_config
    keyterms: list[dict[str, Any]] | None = (
        Field(  # Changed from 'keywords' to 'keyterms'
            default=None,
            description="Extracted keyterms.",  # Changed description
        )
    )

    def __init__(self, **data):
        """Initialize the KeyTerms class, ensuring a tokenizer is set.

        If a tokenizer is not provided, creates one using the specified spaCy model.
        """
        if "tokenizer" not in data or data["tokenizer"] is None:
            data["tokenizer"] = Tokenizer(model=data.get("model", "xx_sent_ud_sm"))
        super().__init__(**data)

    def __call__(self) -> dict:
        """Extract keyterms from the document using the specified method.

        Returns:
            dict: Dictionary containing extracted keyterms and their scores.
        """
        if isinstance(self.document, Doc):
            doc = self.document
        elif isinstance(self.document, str):
            doc = self.tokenizer(self.document)
        else:
            raise ValueError("The 'document' field must be a string or a spaCy Doc.")

        min_n, max_n = self.ngrams

        if self.method == "textrank":
            results = extract.keyterms.textrank(
                doc,
                normalize=self.normalize,  # <-- Use user-supplied normalize
                topn=self.topn * 20,
            )
            results = [
                (term, score)
                for term, score in results
                if min_n <= len(term.split()) <= max_n
                and term.lower() not in STOP_WORDS
            ][: self.topn]
        elif self.method == "sgrank":
            results = extract.keyterms.sgrank(
                doc,
                normalize=self.normalize,  # <-- Use user-supplied normalize
                ngrams=self.ngrams,
                topn=self.topn,
            )
        else:
            raise ValueError("Invalid method. Choose 'textrank' or 'sgrank'.")

        self.keyterms = [
            {"term": term, "score": score} for term, score in results
        ]  # Changed self.keywords to self.keyterms
        doc._.keyterms = self.keyterms  # Changed doc._.keywords to doc._.keyterms
        return self.to_dict()

    def to_dict(self):
        """Return the extracted keyterms as a dictionary."""  # Changed description
        return {
            "keyterms": [  # Changed "keywords" to "keyterms"
                {"term": kw["term"], "score": kw["score"]}
                for kw in (
                    self.keyterms or []
                )  # Changed self.keywords to self.keyterms
            ]
        }

    def to_df(self):
        """Return the extracted keyterms as a pandas DataFrame."""  # Changed description
        return pd.DataFrame(
            getattr(self, "keyterms", [])
        )  # Changed "keywords" to "keyterms"

    def to_list(self):
        """Return the extracted keyterms as a list of (term, score) tuples."""  # Changed description
        return [
            (kw["term"], kw["score"]) for kw in getattr(self, "keyterms", [])
        ]  # Changed "keywords" to "keyterms"
