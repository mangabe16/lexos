from typing import Optional, Literal, List, Dict, Any, Tuple
# ... rest of your imports

class TextacyKeywords(BaseModel):
    """Extracts keywords from text using textacy algorithms."""

    text: str = Field(..., description="The raw text to analyze.")
    method: Literal["textrank", "sgrank"] = Field(
        "textrank", description="The keyword extraction method."
    )
    topn: int = Field(10, gt=0, description="Number of top keywords to return.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self) -> Dict[str, List[Dict[str, Any]]]:
        doc = textacy.make_spacy_doc(self.text, lang="en_core_web_sm")

        if self.method == "textrank":
            results: List[Tuple[str, float]] = extract.keyterms.textrank(doc, normalize="lemma", topn=self.topn)
        elif self.method == "sgrank":
            results: List[Tuple[str, float]] = extract.keyterms.sgrank(
                doc, normalize="lower", ngrams=(1, 2, 3), topn=self.topn
            )
        else:
            raise ValueError("Unsupported keyword extraction method.")

        return {"keywords": [{"term": kw, "score": score} for kw, score in results]}

class ZTestTopwords(BaseModel):
    """Finds topwords using a Z-test between target and background documents."""

    target_texts: List[str] = Field(..., description="Target document(s).")
    background_texts: List[str] = Field(..., description="Background document(s).")
    topn: int = Field(10, gt=0, description="Number of topwords to return.")

    def __call__(self) -> Dict[str, List[Dict[str, Any]]]:
        # Tokenize and count terms
        tokenizer: Tokenizer = Tokenizer(model="en_core_web_sm")
        target_docs: List[Any] = list(tokenizer.make_docs(self.target_texts))
        background_docs: List[Any] = list(tokenizer.make_docs(self.background_texts))

        def get_tokens(docs: List[Any]) -> List[str]:
            # Filter tokens: not stopwords, not punctuation, not space
            return [
                token.lemma_.lower()
                for doc in docs
                for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
            ]

        target_tokens: List[str] = get_tokens(target_docs)
        background_tokens: List[str] = get_tokens(background_docs)

        target_counts: Counter = Counter(target_tokens)
        background_counts: Counter = Counter(background_tokens)

        # Calculate frequencies
        target_total: int = sum(target_counts.values())
        background_total: int = sum(background_counts.values())

        results: List[Tuple[str, float]] = []
        all_terms: set = set(target_counts) | set(background_counts)
        for term in all_terms:
            p1: float = target_counts[term] / target_total if target_total else 0
            p2: float = background_counts[term] / background_total if background_total else 0
            p: float = (target_counts[term] + background_counts[term]) / (target_total + background_total) if (target_total + background_total) else 0
            n1, n2 = target_total, background_total
            if n1 > 0 and n2 > 0 and p > 0 and p < 1:
                z: float = (p1 - p2) / np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
                # Optionally add p-value:
                # p_value = 2 * (1 - norm.cdf(abs(z)))
                results.append((term, z))
        # Sort by absolute z-score
        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return {"topwords": [{"term": t, "zscore": z} for t, z in results[:self.topn]]}
