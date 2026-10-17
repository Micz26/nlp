from pathlib import Path

import pandas as pd

from src.annotation_common import LABELS, cohen_kappa, fleiss_kappa, load_documents


def test_document_pages_are_merged() -> None:
    documents = load_documents()

    assert len(documents) == 443
    assert documents["filename"].is_unique
    assert documents["page_count"].sum() == 1093


def test_cohen_kappa_is_one_for_identical_ratings() -> None:
    ratings = ["medical_doc", "official_doc", "other"]

    assert cohen_kappa(ratings, ratings) == 1.0


def test_fleiss_kappa_is_one_for_identical_ratings() -> None:
    rows = pd.DataFrame(
        [
            {"task_id": task_id, "annotator": annotator, "label": label}
            for task_id, label in enumerate(LABELS, start=1)
            for annotator in ("A", "B")
        ]
    )

    assert fleiss_kappa(rows) == 1.0


def test_generated_final_dataset_is_complete() -> None:
    output = Path("data/annotated/corpus_medical_official_docs_annotated.parquet")
    assert output.exists()
    final = pd.read_parquet(output)

    assert len(final) == 1093
    assert not final.duplicated(["id", "page_number"]).any()
    assert not final["label"].isna().any()
    assert set(final["label"]).issubset(LABELS)


def test_versioned_row_labels_cover_the_source_dataset() -> None:
    source = pd.read_parquet("data/corpus_medical_official_docs.parquet", columns=["id"])
    labels = pd.read_csv("annotations/row_labels.csv", dtype={"row_id": "string"})

    assert list(labels.columns) == ["row_id", "label"]
    assert labels["row_id"].is_unique
    assert len(labels) == 443
    assert set(labels["row_id"]) == set(source["id"].astype("string"))
