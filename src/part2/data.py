from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from src.annotation_common import LABELS, ROOT, load_documents


CACHE_DIR = ROOT / "data" / "part2"
FIGURES_DIR = ROOT / "reports" / "figures"
ANNOTATED_DOCUMENTS = ROOT / "annotations" / "annotated_documents.csv"
TEMPLATE_DISTANCE_THRESHOLD = 0.5
WORD_PATTERN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
FORM_CODE_PATTERN = re.compile(
    r"\b(OL-\d+A?|PR-\d+|Z-3|PIT-\d+[A-Z]?|SR-1|RPD-?\w*|CRP|KEP|POLTAX)\b"
)


def clean_text(text: str) -> str:
    """Removes OCR/markdown layout noise while keeping the running text intact."""
    text = text.replace("<br>", " ")
    text = re.sub(r"\|?\s*:?-{3,}:?\s*", " ", text)
    text = re.sub(r"[|*#_]+", " ", text)
    text = re.sub(r"([.…\-–=~])\1{2,}", " ", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_PATTERN.findall(text)]


def template_skeleton(text: str) -> str:
    """Form layout without filled-in values: first part of the document, no digits."""
    skeleton = re.sub(r"\d+", " ", text[:3000])
    skeleton = re.sub(r"[|*:\-#<>]+|br", " ", skeleton)
    return re.sub(r"\s+", " ", skeleton).lower()


def assign_templates(texts: pd.Series) -> np.ndarray:
    features = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(
        texts.map(template_skeleton)
    )
    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=TEMPLATE_DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="average",
    ).fit_predict(features.toarray())


def template_names(corpus: pd.DataFrame) -> dict[int, str]:
    names = {}
    for template_id, group in corpus.groupby("template_id"):
        codes = group["clean_text"].str[:600].str.extract(FORM_CODE_PATTERN)[0].dropna()
        code = codes.mode().iloc[0] if not codes.empty else ""
        words = tokenize(group["clean_text"].iloc[0][:400])
        stop = {"zus", "zakład", "ubezpieczeń", "społecznych", "nr", "do", "umowy", "załącznik"}
        words = [word for word in words if word not in stop and not re.fullmatch(r"[a-f]{1,8}", word)]
        title = " ".join(words[:5])
        names[template_id] = f"T{template_id:02d} {code} {title}".replace("  ", " ")
    return names


def load_corpus(cache: bool = True) -> pd.DataFrame:
    """One row per document with final label, annotation metadata and template group."""
    path = CACHE_DIR / "corpus.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)

    documents = load_documents()
    annotations = pd.read_csv(ANNOTATED_DOCUMENTS)
    corpus = documents.merge(
        annotations[
            ["filename", "final_label", "phase", "round", "annotator", "reason_short", "had_disagreement"]
        ],
        on="filename",
        how="left",
        validate="one_to_one",
    ).rename(columns={"final_label": "label"})
    if corpus["label"].isna().any() or not set(corpus["label"]).issubset(LABELS):
        raise ValueError("Every document needs exactly one known label")

    corpus["clean_text"] = corpus["text"].map(clean_text)
    corpus["clean_character_count"] = corpus["clean_text"].str.len()
    corpus["word_count"] = corpus["clean_text"].map(lambda text: len(tokenize(text)))
    corpus["template_id"] = assign_templates(corpus["clean_text"])
    corpus["template"] = corpus["template_id"].map(template_names(corpus))
    corpus["snippet"] = corpus["clean_text"].str[:400].str.replace("\n", " ", regex=False)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(path, index=False)
    return corpus
