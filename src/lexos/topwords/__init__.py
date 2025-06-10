"""__init__.py.

Last Update: 10 June, 2025
Last Tested: 10 June, 2025

Credits:

    See topwords Lexos web app model:
    https://github.com/WheatonCS/Lexos/blob/master/lexos/models/top_words_model.py
    Developed by Hanna Ondrasek & Gabe Albernaz
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc
import textacy
from textacy import extract

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=doc_schema
)


class TextacyKeywords(BaseModel):
    """Extracts keywords from text using textacy algorithms."""

    text: str = Field(..., description="The raw text to analyze.")
    method: Literal["textrank", "sgrank"] = Field(
        "textrank", description="The keyword extraction method."
    )
    topn: int = Field(10, gt=0, description="Number of top keywords to return.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self) -> dict:
        doc = textacy.make_spacy_doc(self.text, lang="en_core_web_sm")

        if self.method == "textrank":
            results = extract.keyterms.textrank(doc, normalize="lemma", topn=self.topn)
        elif self.method == "sgrank":
            results = extract.keyterms.sgrank(
                doc, normalize="lower", ngrams=(1, 2, 3), topn=self.topn
            )
        else:
            raise ValueError("Unsupported keyword extraction method.")

        return {"keywords": [{"term": kw, "score": score} for kw, score in results]}
