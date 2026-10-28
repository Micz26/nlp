# NLP

Repozytorium na projekt z NLP.

## Dane

Pobierz korpus z [folderu na Google Drive](https://drive.google.com/drive/folders/1Y_ZmpGe1qKeGmk8WvSpZdChvmTB6Xkt4?usp=drive_link) i zapisz jako `data/corpus_medical_official_docs.parquet`.

Pliki w `data/` (poza `data/scripts/`) są ignorowane przez Git.

## Uruchomienie

```bash
uv sync
uv run python src/compile_independent_annotations.py
uv run python data/scripts/build_annotated_dataset.py
uv run pytest
```

Wynik trafia do `data/annotated/corpus_medical_official_docs_annotated.parquet`.

## Label Studio

```bash
LABEL_STUDIO_BASE_DATA_DIR=.label-studio uv run label-studio start
```

Konfiguracja projektu: `config/label_studio.xml`.

## Wyniki

Raport i statystyki: `annotations/REPORT.md`.
