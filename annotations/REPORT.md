# Raport z anotacji korpusu

## Zakres i organizacja

Raport obejmuje klasyfikację całych dokumentów do klas `medical_doc`, `official_doc` i `other`.

Anotację wykonały trzy niezależne subagenty AI działające w rolach Mikołaja, Szymona i Michała.

Mikołaj i Szymon niezależnie ocenili dwie rozłączne próby pilotażowe po 40 dokumentów.

Michał rozstrzygnął przypadki sporne zgodnie z procedurą 2+1.

Po drugiej iteracji pozostałe 363 dokumenty podzielono losowo i bez nakładania po 121 dokumentów na każdego anotatora.

## Przygotowanie danych

Plik wejściowy zawierał 1093 stron należących do 443 dokumentów.

Strony połączono według nazwy pliku i uporządkowano według numeru strony.

Puste pole tekstowe na pojedynczej stronie zastąpiono pustym ciągiem, nie usuwając całego dokumentu.

Mediana długości dokumentu wynosi 4637 znaków, średnia 10506.4, minimum 2215, a maksimum 392106.

Mediana liczby stron wynosi 2, średnia 2.47, minimum 2, a maksimum 5.

## Zgodność anotatorów

| Iteracja | Dokumenty | Zgodne | Sporne | Zgodność | Kappa Cohena | Kappa Fleissa |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 38 | 2 | 0.950 | 0.916 | 0.916 |
| 2 | 40 | 40 | 0 | 1.000 | 1.000 | 1.000 |

Po pierwszej iteracji doprecyzowano sposób traktowania formularzy ZUS, dokumentów kontroli placówek medycznych, ankiet pacjentów i list ergonomii stanowiska pracy.

Kappa Cohena wzrosła z 0.916 do 1.000, a Kappa Fleissa z 0.916 do 1.000.

W pierwszej rundzie wystąpiły 2 spory, a w drugiej 0.

Michał rozstrzygnął 2 sporne dokumenty na podstawie pełnej treści i drugiej wersji wytycznych.

## Końcowy zbiór

| Etykieta | Liczba | Udział |
|---|---:|---:|
| `medical_doc` | 130 | 29.3% |
| `official_doc` | 285 | 64.3% |
| `other` | 28 | 6.3% |

Łącznie zaanotowano 443 z 443 dokumentów i każdy dokument ma dokładnie jedną etykietę końcową.

W części produkcyjnej Mikołaj zaanotował 121 dokumentów, Szymon 121, a Michał 121.

## Artefakty

Konfiguracja interfejsu znajduje się w `config/label_studio.xml`.

Plik `data/label_studio/tasks.json` można zaimportować do projektu Label Studio.

Plik `data/label_studio/export.json` zawiera decyzje anotatorów w formacie eksportowym Label Studio.

Surowe decyzje zachowano w katalogu `annotations/independent` oraz w `pilot_annotations.csv`.

Końcowe etykiety dokumentów znajdują się w `annotated_documents.csv`, a mapowanie wierszy źródłowych w `row_labels.csv`.

Skrypt `data/scripts/build_annotated_dataset.py` łączy mapowanie z oryginalnym korpusem i zapisuje pełny wynik w ignorowanym przez Git katalogu `data/annotated`.
