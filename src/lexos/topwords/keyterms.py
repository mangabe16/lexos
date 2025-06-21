from lexos.topwords import TopWords
from spacy.tokens import Doc
from pydantic import Field, ConfigDict
from lexos.tokenizer import Tokenizer
from textacy import extract

# register a custom extension for keywords if not already set
if not Doc.has_extension("keywords"):
    Doc.set_extension("keywords", default=None, force=True)

class KeyTerms(TopWords):
    """Extracts keywords from text or a spaCy Doc using textacy algorithms."""

    document: str | Doc | None = Field(None, description="The raw text or spaCy doc to analyze.")
    method: Literal["textrank", "sgrank"] = Field(
        ..., description="Method for keyword extraction (e.g., 'textrank', 'sgrank')."
    )
    topn: int = Field(
        10, gt=0, description="Number of top keywords to return."
    )
    model: str = Field(
        default="en_core_web_sm",
        description="spaCy model name to use for tokenization."
    )
    ngrams: Tuple[int, int] = Field(
        default=(1, 3),
        description="The ngram range for keyword extraction, e.g., (1, 1) for unigrams only."
    )
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)
    normalize: str | None = Field(
        default=None,
        description="Normalization for keyterm extraction (e.g., 'lemma', 'lower', or None)."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
    keywords: list[dict[str, Any]] | None = Field(
        default=None, description="Extracted keywords."
    )

    def __init__(self, **data):
        if "tokenizer" not in data or data["tokenizer"] is None:
            data["tokenizer"] = Tokenizer(model=data.get("model", "xx_sent_ud_sm"))
        super().__init__(**data)

    def __call__(self) -> dict:
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
                topn=self.topn * 20
            )
            results = [
                (term, score)
                for term, score in results
                if min_n <= len(term.split()) <= max_n and term.lower() not in STOP_WORDS
            ][:self.topn]
        elif self.method == "sgrank":
            results = extract.keyterms.sgrank(
                doc,
                normalize=self.normalize,  # <-- Use user-supplied normalize
                ngrams=self.ngrams,
                topn=self.topn
            )
        else:
            raise ValueError("Invalid method. Choose 'textrank' or 'sgrank'.")

        self.keywords = [
            {"term": term, "score": score} for term, score in results
        ]
        doc._.keywords = self.keywords
        return self.to_dict()

    def to_dict(self):
        return {
            "keywords": [
                {"term": kw["term"], "score": kw["score"]}
                for kw in (self.keywords or [])
            ]
        }

    def to_df(self):
        return pd.DataFrame(getattr(self, "keywords", []))

    def to_list(self):
        return [(kw["term"], kw["score"]) for kw in getattr(self, "keywords", [])]

