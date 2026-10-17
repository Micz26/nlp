from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "corpus_medical_official_docs.parquet"
DEFAULT_LABELS = ROOT / "annotations" / "row_labels.csv"
DEFAULT_OUTPUT = ROOT / "data" / "annotated" / "corpus_medical_official_docs_annotated.parquet"
ALLOWED_LABELS = {"medical_doc", "official_doc", "other"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join versioned row labels with the original page-level corpus."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build(input_path: Path, labels_path: Path, output_path: Path) -> pd.DataFrame:
    source = pd.read_parquet(input_path)
    labels = pd.read_csv(labels_path, dtype={"row_id": "string", "label": "string"})

    if "id" not in source.columns:
        raise ValueError("Input dataset has no 'id' column")
    if list(labels.columns) != ["row_id", "label"]:
        raise ValueError("Labels file must contain exactly: row_id,label")
    if labels.row_id.duplicated().any():
        raise ValueError("Label row identifiers must be unique")
    if labels.isna().any().any() or not set(labels.label).issubset(ALLOWED_LABELS):
        raise ValueError("Labels file contains missing or unknown labels")

    source = source.assign(id=source.id.astype("string"))
    source_ids = set(source.id)
    label_ids = set(labels.row_id)
    if source_ids != label_ids:
        missing = len(source_ids - label_ids)
        extra = len(label_ids - source_ids)
        raise ValueError(f"Row ID mismatch: {missing} missing labels and {extra} extra labels")

    annotated = source.merge(
        labels,
        left_on="id",
        right_on="row_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="row_id")
    if annotated.label.isna().any() or len(annotated) != len(source):
        raise AssertionError("Annotated output is incomplete")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_parquet(output_path, index=False)
    return annotated


def main() -> None:
    args = parse_args()
    annotated = build(args.input, args.labels, args.output)
    print(f"Wrote {len(annotated)} annotated rows to {args.output}")
    print(annotated.label.value_counts().to_string())


if __name__ == "__main__":
    main()
