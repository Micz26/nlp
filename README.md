# Anotacja dokumentów medycznych i urzędowych

Repozytorium zawiera odtwarzalny proces realizacji Części I zadania z pliku `listy/Zadanie_1.pdf`.

Korpus jest anotowany na poziomie całego dokumentu w klasach `medical_doc`, `official_doc` i `other`.

Proces obejmuje dwie iteracje pomiaru zgodności, aktualizację wytycznych, rozstrzyganie sporów w procedurze 2+1 oraz anotację całego pozostałego zbioru.

## Dane wejściowe

Pełne teksty dokumentów nie są przechowywane w repozytorium ze względu na występujące w nich dane wyglądające jak dane osobowe.

Zbiór należy pobrać z [folderu udostępnionego do zadania](https://drive.google.com/drive/folders/1Y_ZmpGe1qKeGmk8WvSpZdChvmTB6Xkt4?usp=drive_link) i zapisać jako `data/corpus_medical_official_docs.parquet`.

Oczekiwany plik ma 1093 wiersze i kolumny `id`, `filename`, `page_number` oraz `page_text`.

Suma SHA-256 użytego pliku wynosi `384e5e7300f2a427965c74262feae4781b7bf00fe61e9acb44c27c918d43c5e2`.

Pliki danych w `data/` oraz katalog `listy/` są ignorowane przez Git.

Wyjątkiem są wersjonowane skrypty w `data/scripts/`.

Ignorowanych plików danych nie należy dodawać z użyciem `git add -f`.

## Uruchomienie

1. Pobierz korpus z podanego wyżej folderu Google Drive.
2. Umieść plik pod dokładną ścieżką `data/corpus_medical_official_docs.parquet`.
3. Zainstaluj zależności projektu.
4. Skompiluj wyniki anotacji i zbuduj lokalny zaanotowany zbiór.
5. Uruchom testy sprawdzające kompletność danych.

```bash
uv sync
uv run python src/compile_independent_annotations.py
uv run python data/scripts/build_annotated_dataset.py
uv run pytest
```

Gotowy zbiór zostanie zapisany jako `data/annotated/corpus_medical_official_docs_annotated.parquet`.

Katalog `data/annotated/` zostanie utworzony automatycznie.

Jeżeli katalog cache `uv` nie jest zapisywalny, można uruchomić polecenia z `UV_CACHE_DIR=/tmp/uv-cache`.

## Label Studio

Zależność `label-studio` jest zapisana w `pyproject.toml` i `uv.lock`.

Interfejs można uruchomić poleceniem:

```bash
LABEL_STUDIO_BASE_DATA_DIR=.label-studio uv run label-studio start
```

Po utworzeniu projektu należy wkleić konfigurację z `config/label_studio.xml` i zaimportować `data/label_studio/tasks.json`.

Kompletny eksport z decyzjami anotatorów znajduje się w `data/label_studio/export.json`.

Oba pliki zawierają pełne teksty dokumentów, dlatego pozostają lokalnie w ignorowanym katalogu `data/`.

## Wyniki

Opis procesu i statystyki znajdują się w `annotations/REPORT.md`.

Wersjonowane mapowanie etykiet znajduje się w `annotations/row_labels.csv` i ma wyłącznie kolumny `row_id,label`.

Każdy `row_id` odpowiada jednemu dokumentowi i może wystąpić na kilku stronach oryginalnego zbioru.

Skrypt `data/scripts/build_annotated_dataset.py` łączy je z oryginalnym zbiorem i zapisuje wynik w `data/annotated/corpus_medical_official_docs_annotated.parquet`.

Indeks dokumentów wraz z końcowymi etykietami znajduje się w `annotations/annotated_documents.csv`.

Anotacje wykonały trzy niezależne subagenty AI działające w rolach Mikołaja, Szymona i Michała.

Do repozytorium można bezpiecznie dodać raport, metryki, wytyczne, indeks CSV oraz pliki z decyzjami anotatorów, ponieważ nie zawierają pełnej treści dokumentów.

Pełny zaanotowany Parquet oraz eksporty Label Studio znajdują się pod `data/` i są ignorowane przez Git.

Skrypt kompilujący odtwarza je lokalnie po pobraniu danych wejściowych.

## Część II: osadzenia i klasyfikacja

Część II jest w notebooku `notebooks/part2_embeddings_classification.ipynb`.
Jego źródło w formacie jupytext to `notebooks/part2_embeddings_classification.py`, a moduły pomocnicze leżą w `src/part2/`.
Zadania dotyczące dokumentów pisanych odręcznie zostały pominięte.

| Moduł | Zawartość |
|---|---|
| `src/part2/data.py` | łączenie stron, czyszczenie szumu OCR, grupowanie dokumentów w szablony formularzy |
| `src/part2/lemmatize.py` | lematyzacja spaCy `pl_core_news_lg` |
| `src/part2/embeddings.py` | Word2Vec i fastText KGR10 (średnia, SIF) oraz enkodery `sdadas/mmlw-roberta-base` i `BAAI/bge-m3` (strategie head / chunks / long) |
| `src/part2/analysis.py` | UMAP i t-SNE (plotly), miary separacji, sąsiedzi, HDBSCAN |
| `src/part2/classify.py` | regresja logistyczna, powtarzana stratyfikowana CV, leave-template-out, reguły z wytycznych |

```bash
uv sync
uv run python src/compile_independent_annotations.py
uv run python data/scripts/build_annotated_dataset.py
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 part2_embeddings_classification.ipynb
```

Pierwsze uruchomienie pobiera z Hugging Face około 10 GB modeli i trwa około 20 minut na Apple M2 Pro.
Wyniki pośrednie są cache'owane w `data/part2/`, a interaktywne wykresy HTML i macierze pomyłek trafiają do `reports/figures/`.
Pliki HTML w `reports/figures/` zawierają fragmenty tekstów dokumentów, dlatego są ignorowane przez Git i powstają lokalnie po uruchomieniu notebooka.
