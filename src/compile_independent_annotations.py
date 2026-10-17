from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from annotation_common import (
    ANNOTATORS,
    INPUT_PATH,
    LABELS,
    OUTPUT_DIR,
    fleiss_kappa,
    label_studio_result,
    load_documents,
    round_metrics,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT_DIR = OUTPUT_DIR / "independent"
MANIFEST_PATH = OUTPUT_DIR / "assignment_manifest.csv"
LABEL_STUDIO_DIR = ROOT / "data" / "label_studio"


def read_annotations(path: Path, expected_annotator: str) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = ["task_id", "filename", "annotator", "label", "reason_short"]
    if set(rows.columns) != set(required):
        raise ValueError(f"Unexpected columns in {path}: {rows.columns.tolist()}")
    if set(rows.annotator) != {expected_annotator}:
        raise ValueError(f"Unexpected annotator in {path}")
    if rows.task_id.duplicated().any() or rows[required].isna().any().any():
        raise ValueError(f"Duplicates or missing values in {path}")
    if not set(rows.label).issubset(LABELS):
        raise ValueError(f"Unexpected label in {path}")
    return rows


def validate_ids(rows: pd.DataFrame, expected_ids: set[int], description: str) -> None:
    actual_ids = set(rows.task_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"Wrong IDs for {description}; missing={missing}, extra={extra}")


def pair_rows(left: pd.DataFrame, right: pd.DataFrame, round_number: int) -> pd.DataFrame:
    columns = ["task_id", "filename", "annotator", "label", "reason_short"]
    return pd.concat([left[columns], right[columns]], ignore_index=True).assign(
        round=round_number,
        independent=True,
        guideline_version=round_number,
    )


def compile_pilots(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict]]:
    pilot_1_ids = set(manifest.loc[manifest["round"] == 1, "task_id"])
    pilot_2_ids = set(manifest.loc[manifest["round"] == 2, "task_id"])

    m1 = read_annotations(INDEPENDENT_DIR / "mikolaj_round1_v1.csv", "Mikołaj")
    s1 = read_annotations(INDEPENDENT_DIR / "szymon_round1_v1.csv", "Szymon")
    validate_ids(m1, pilot_1_ids, "Mikołaj, round 1")
    validate_ids(s1, pilot_1_ids, "Szymon, round 1")

    m_all = read_annotations(INDEPENDENT_DIR / "mikolaj.csv", "Mikołaj")
    s_all = read_annotations(INDEPENDENT_DIR / "szymon.csv", "Szymon")
    m2 = m_all[m_all.task_id.isin(pilot_2_ids)]
    s2 = s_all[s_all.task_id.isin(pilot_2_ids)]
    validate_ids(m2, pilot_2_ids, "Mikołaj, round 2")
    validate_ids(s2, pilot_2_ids, "Szymon, round 2")

    rounds = pd.concat([pair_rows(m1, s1, 1), pair_rows(m2, s2, 2)], ignore_index=True)
    metrics = {number: round_metrics(group) for number, group in rounds.groupby("round")}

    adjudication = read_annotations(INDEPENDENT_DIR / "michal_adjudication.csv", "Michał")
    pivots = rounds.pivot(index=["round", "task_id", "filename"], columns="annotator", values="label")
    disputed = pivots[pivots["Mikołaj"] != pivots["Szymon"]].reset_index()
    validate_ids(adjudication, set(disputed.task_id), "Michał, adjudication")
    adjudication_by_id = adjudication.set_index("task_id")

    consensus_rows = []
    for row in pivots.reset_index().itertuples():
        disagreement = row.Mikołaj != row.Szymon
        final_label = adjudication_by_id.loc[row.task_id, "label"] if disagreement else row.Mikołaj
        consensus_rows.append(
            {
                "task_id": row.task_id,
                "filename": row.filename,
                "round": row.round,
                "final_label": final_label,
                "had_disagreement": disagreement,
                "adjudicator": "Michał" if disagreement else "",
            }
        )
    return rounds, pd.DataFrame(consensus_rows), metrics


def compile_production(manifest: pd.DataFrame) -> pd.DataFrame:
    all_rows = {
        "Mikołaj": read_annotations(INDEPENDENT_DIR / "mikolaj.csv", "Mikołaj"),
        "Szymon": read_annotations(INDEPENDENT_DIR / "szymon.csv", "Szymon"),
        "Michał": read_annotations(INDEPENDENT_DIR / "michal.csv", "Michał"),
    }
    expected = manifest[manifest.phase == "production"]
    result = []
    for annotator in ANNOTATORS:
        ids = set(expected.loc[expected.assigned_annotator == annotator, "task_id"])
        selected = all_rows[annotator][all_rows[annotator].task_id.isin(ids)].copy()
        validate_ids(selected, ids, f"{annotator}, production")
        selected = selected.rename(columns={"label": "final_label"})
        selected["phase"] = "production"
        selected["round"] = 0
        selected["had_disagreement"] = False
        selected["adjudicator"] = ""
        result.append(selected)
    return pd.concat(result, ignore_index=True)


def build_report(
    documents: pd.DataFrame,
    rounds: pd.DataFrame,
    consensus: pd.DataFrame,
    production: pd.DataFrame,
    final: pd.DataFrame,
    metrics: dict[int, dict],
) -> str:
    distribution = final.final_label.value_counts().reindex(LABELS, fill_value=0)
    assignment = production.annotator.value_counts().reindex(ANNOTATORS, fill_value=0)
    chars = documents.character_count
    pages = documents.page_count
    disputes = consensus[consensus.had_disagreement]

    def metric_line(number: int) -> str:
        item = metrics[number]
        return (
            f"| {number} | {item['documents']} | {item['agreements']} | "
            f"{item['disagreements']} | {item['percent_agreement']:.3f} | "
            f"{item['cohen_kappa']:.3f} | {item['fleiss_kappa']:.3f} |"
        )

    return f"""# Raport z anotacji korpusu

## Zakres i organizacja

Raport obejmuje klasyfikację całych dokumentów do klas `medical_doc`, `official_doc` i `other`.

Anotację wykonały trzy niezależne subagenty AI działające w rolach Mikołaja, Szymona i Michała.

Mikołaj i Szymon niezależnie ocenili dwie rozłączne próby pilotażowe po 40 dokumentów.

Michał rozstrzygnął przypadki sporne zgodnie z procedurą 2+1.

Po drugiej iteracji pozostałe 363 dokumenty podzielono losowo i bez nakładania po 121 dokumentów na każdego anotatora.

## Przygotowanie danych

Plik wejściowy zawierał {documents.page_count.sum()} stron należących do {len(documents)} dokumentów.

Strony połączono według nazwy pliku i uporządkowano według numeru strony.

Puste pole tekstowe na pojedynczej stronie zastąpiono pustym ciągiem, nie usuwając całego dokumentu.

Mediana długości dokumentu wynosi {chars.median():.0f} znaków, średnia {chars.mean():.1f}, minimum {chars.min()}, a maksimum {chars.max()}.

Mediana liczby stron wynosi {pages.median():.0f}, średnia {pages.mean():.2f}, minimum {pages.min()}, a maksimum {pages.max()}.

## Zgodność anotatorów

| Iteracja | Dokumenty | Zgodne | Sporne | Zgodność | Kappa Cohena | Kappa Fleissa |
|---:|---:|---:|---:|---:|---:|---:|
{metric_line(1)}
{metric_line(2)}

Po pierwszej iteracji doprecyzowano sposób traktowania formularzy ZUS, dokumentów kontroli placówek medycznych, ankiet pacjentów i list ergonomii stanowiska pracy.

Kappa Cohena wzrosła z {metrics[1]['cohen_kappa']:.3f} do {metrics[2]['cohen_kappa']:.3f}, a Kappa Fleissa z {metrics[1]['fleiss_kappa']:.3f} do {metrics[2]['fleiss_kappa']:.3f}.

W pierwszej rundzie wystąpiły {metrics[1]['disagreements']} spory, a w drugiej {metrics[2]['disagreements']}.

Michał rozstrzygnął {len(disputes)} sporne dokumenty na podstawie pełnej treści i drugiej wersji wytycznych.

## Końcowy zbiór

| Etykieta | Liczba | Udział |
|---|---:|---:|
{chr(10).join(f'| `{label}` | {int(distribution[label])} | {distribution[label] / len(final):.1%} |' for label in LABELS)}

Łącznie zaanotowano {len(final)} z {len(documents)} dokumentów i każdy dokument ma dokładnie jedną etykietę końcową.

W części produkcyjnej Mikołaj zaanotował {assignment['Mikołaj']} dokumentów, Szymon {assignment['Szymon']}, a Michał {assignment['Michał']}.

## Artefakty

Konfiguracja interfejsu znajduje się w `config/label_studio.xml`.

Plik `data/label_studio/tasks.json` można zaimportować do projektu Label Studio.

Plik `data/label_studio/export.json` zawiera decyzje anotatorów w formacie eksportowym Label Studio.

Surowe decyzje zachowano w katalogu `annotations/independent` oraz w `pilot_annotations.csv`.

Końcowe etykiety dokumentów znajdują się w `annotated_documents.csv`, a mapowanie wierszy źródłowych w `row_labels.csv`.

Skrypt `data/scripts/build_annotated_dataset.py` łączy mapowanie z oryginalnym korpusem i zapisuje pełny wynik w ignorowanym przez Git katalogu `data/annotated`.
"""


def main() -> None:
    documents = load_documents()
    manifest = pd.read_csv(MANIFEST_PATH)
    validate_ids(manifest, set(documents.task_id), "assignment manifest")
    rounds, consensus, metrics = compile_pilots(manifest)
    production = compile_production(manifest)

    pilot_final = consensus.assign(
        phase="pilot",
        annotator=consensus.adjudicator.where(consensus.had_disagreement, "Mikołaj + Szymon"),
        reason_short="",
    )
    final_labels = pd.concat(
        [
            pilot_final[
                [
                    "task_id",
                    "final_label",
                    "phase",
                    "round",
                    "annotator",
                    "reason_short",
                    "had_disagreement",
                    "adjudicator",
                ]
            ],
            production[
                [
                    "task_id",
                    "final_label",
                    "phase",
                    "round",
                    "annotator",
                    "reason_short",
                    "had_disagreement",
                    "adjudicator",
                ]
            ],
        ],
        ignore_index=True,
    )
    final = documents.merge(final_labels, on="task_id", validate="one_to_one").sort_values("task_id")
    if len(final) != len(documents) or final.final_label.isna().any():
        raise AssertionError("Every document must have exactly one final label")

    rounds.to_csv(OUTPUT_DIR / "pilot_annotations.csv", index=False)
    consensus.to_csv(OUTPUT_DIR / "pilot_consensus.csv", index=False)
    final.drop(columns=["text", "source_ids"]).to_csv(
        OUTPUT_DIR / "annotated_documents.csv", index=False
    )
    source_rows = pd.read_parquet(INPUT_PATH, columns=["id", "filename"]).drop_duplicates()
    if source_rows.id.duplicated().any() or source_rows.filename.duplicated().any():
        raise AssertionError("Source document IDs and filenames must form a one-to-one mapping")
    row_labels = source_rows.merge(
        final[["filename", "final_label"]], on="filename", how="left", validate="many_to_one"
    )
    if row_labels.id.duplicated().any() or row_labels.final_label.isna().any():
        raise AssertionError("Every unique source row must have exactly one final label")
    row_labels.rename(columns={"id": "row_id", "final_label": "label"})[
        ["row_id", "label"]
    ].to_csv(OUTPUT_DIR / "row_labels.csv", index=False)
    write_json(OUTPUT_DIR / "agreement_metrics.json", metrics)

    raw_by_task = {
        task_id: group.to_dict("records") for task_id, group in rounds.groupby("task_id")
    }
    adjudication = read_annotations(INDEPENDENT_DIR / "michal_adjudication.csv", "Michał")
    adjudication_by_id = adjudication.set_index("task_id")
    production_by_id = production.set_index("task_id")
    export = []
    for row in final.itertuples():
        annotations = []
        for raw in raw_by_task.get(row.task_id, []):
            annotations.append(
                {
                    "completed_by": raw["annotator"],
                    "result": label_studio_result(raw["label"]),
                    "was_cancelled": False,
                    "ground_truth": False,
                }
            )
        if row.task_id in adjudication_by_id.index:
            item = adjudication_by_id.loc[row.task_id]
            annotations.append(
                {
                    "completed_by": "Michał",
                    "result": label_studio_result(item.label),
                    "was_cancelled": False,
                    "ground_truth": True,
                }
            )
        if row.task_id in production_by_id.index:
            item = production_by_id.loc[row.task_id]
            annotations.append(
                {
                    "completed_by": item.annotator,
                    "result": label_studio_result(item.final_label),
                    "was_cancelled": False,
                    "ground_truth": False,
                }
            )
        export.append(
            {
                "id": int(row.task_id),
                "data": {"text": row.text, "filename": row.filename},
                "annotations": annotations,
            }
        )
    LABEL_STUDIO_DIR.mkdir(parents=True, exist_ok=True)
    write_json(LABEL_STUDIO_DIR / "export.json", export)
    tasks = [
        {
            "id": int(row.task_id),
            "data": {"text": row.text, "filename": row.filename},
            "meta": {"page_count": int(row.page_count)},
        }
        for row in documents.itertuples()
    ]
    write_json(LABEL_STUDIO_DIR / "tasks.json", tasks)
    (OUTPUT_DIR / "REPORT.md").write_text(
        build_report(documents, rounds, consensus, production, final, metrics), encoding="utf-8"
    )

    print(f"Compiled {len(final)} independently annotated documents")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(final.final_label.value_counts().to_string())


if __name__ == "__main__":
    main()
