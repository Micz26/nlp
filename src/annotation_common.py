from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "corpus_medical_official_docs.parquet"
OUTPUT_DIR = ROOT / "annotations"
LABELS = ("medical_doc", "official_doc", "other")
ANNOTATORS = ("Mikołaj", "Szymon", "Michał")


def load_documents(path: Path = INPUT_PATH) -> pd.DataFrame:
    pages = pd.read_parquet(path).sort_values(["filename", "page_number"])
    pages["page_text"] = pages["page_text"].fillna("")
    documents = (
        pages.groupby("filename", as_index=False)
        .agg(
            source_ids=("id", lambda values: json.dumps(list(values), ensure_ascii=False)),
            page_count=("page_number", "size"),
            text=("page_text", lambda values: "\n\n".join(values)),
        )
        .sort_values("filename")
        .reset_index(drop=True)
    )
    documents.insert(0, "task_id", range(1, len(documents) + 1))
    documents["character_count"] = documents["text"].str.len()
    return documents


def cohen_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Cohen's kappa requires two non-empty, equally sized ratings")
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[label] * right_counts[label] for label in LABELS
    ) / len(left) ** 2
    return (observed - expected) / (1 - expected) if expected != 1 else 1.0


def fleiss_kappa(rows: pd.DataFrame) -> float:
    table = rows.pivot_table(
        index="task_id", columns="label", values="annotator", aggfunc="count", fill_value=0
    ).reindex(columns=LABELS, fill_value=0)
    ratings_per_item = table.sum(axis=1)
    if ratings_per_item.nunique() != 1 or ratings_per_item.iloc[0] < 2:
        raise ValueError("Fleiss' kappa requires the same number of ratings per item")
    n = int(ratings_per_item.iloc[0])
    agreement_per_item = ((table**2).sum(axis=1) - n) / (n * (n - 1))
    observed = float(agreement_per_item.mean())
    category_proportions = table.sum(axis=0) / (len(table) * n)
    expected = float((category_proportions**2).sum())
    return (observed - expected) / (1 - expected) if expected != 1 else 1.0


def round_metrics(rows: pd.DataFrame) -> dict[str, float | int]:
    pivot = rows.pivot(index="task_id", columns="annotator", values="label")
    left = pivot[ANNOTATORS[0]].tolist()
    right = pivot[ANNOTATORS[1]].tolist()
    agreements = sum(a == b for a, b in zip(left, right, strict=True))
    return {
        "documents": len(pivot),
        "agreements": agreements,
        "disagreements": len(pivot) - agreements,
        "percent_agreement": agreements / len(pivot),
        "cohen_kappa": cohen_kappa(left, right),
        "fleiss_kappa": fleiss_kappa(rows),
    }


def label_studio_result(label: str) -> list[dict[str, object]]:
    return [
        {
            "from_name": "label",
            "to_name": "text",
            "type": "choices",
            "value": {"choices": [label]},
        }
    ]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
