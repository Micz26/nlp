from __future__ import annotations

import pandas as pd

from src.part2.data import CACHE_DIR, WORD_PATTERN


SPACY_MODEL = "pl_core_news_lg"


def lemmatize_corpus(corpus: pd.DataFrame, cache: bool = True) -> pd.DataFrame:
    """Returns aligned word forms and spaCy lemmas (alphabetic tokens, lowercased)."""
    path = CACHE_DIR / "lemmas.parquet"
    if cache and path.exists():
        cached = pd.read_parquet(path)
        if cached["filename"].tolist() == corpus["filename"].tolist():
            return cached

    import spacy

    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    nlp.max_length = 2_000_000
    rows = []
    for filename, doc in zip(corpus["filename"], nlp.pipe(corpus["clean_text"], batch_size=16)):
        tokens = [token for token in doc if WORD_PATTERN.fullmatch(token.text)]
        rows.append(
            {
                "filename": filename,
                "forms": [token.text.lower() for token in tokens],
                "lemmas": [token.lemma_.lower() for token in tokens],
            }
        )
    lemmas = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lemmas.to_parquet(path, index=False)
    return lemmas
