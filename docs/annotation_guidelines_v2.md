# Wytyczne anotacji, wersja 2

Jednostką anotacji jest cały dokument po połączeniu wszystkich jego stron w kolejności `page_number`.

Anotator przypisuje dokładnie jedną etykietę na podstawie głównego celu dokumentu i jego nadawcy.

- `medical_doc` - dokument zapisuje stan zdrowia, rozpoznanie, wynik badania, leczenie, rehabilitację, opinię kliniczną, ocenę funkcjonowania pacjenta albo jakość udzielonych świadczeń zdrowotnych.
- `official_doc` - dokument inicjuje, dokumentuje lub rozstrzyga postępowanie urzędowe, podatkowe, ubezpieczeniowe, pracownicze albo kontrolne.
- `other` - dokument nie jest dokumentacją pacjenta ani dokumentem postępowania urzędowego.

## Reguły dla przypadków granicznych

- Formularze ZUS `OL-2`, `OL-4`, `OL-10`, `OL-13`, `PR-4` oraz `Z-3` oznaczamy jako `official_doc`, ponieważ ich głównym celem jest obsługa sprawy ubezpieczeniowej lub administracyjnej.
- Formularz `OL-9`, opinie specjalistyczne, karty rehabilitacji, skale funkcjonowania i skale Barthel oznaczamy jako `medical_doc`, ponieważ zapisują ocenę kliniczną.
- Formularz `OL-9A` oznaczamy jako `medical_doc`, gdy jego treścią jest historia miejsc i przebiegu leczenia.
- Kontrole organizacji świadczeń i obsady placówki oznaczamy jako `official_doc`, nawet gdy dotyczą podmiotu leczniczego.
- Ankiety satysfakcji pacjenta oznaczamy jako `medical_doc`, ponieważ opisują doświadczenie korzystania ze świadczeń zdrowotnych.
- Listy kontrolne ergonomii stanowiska komputerowego bez danych pacjenta oznaczamy jako `other`.
- Sama obecność słów takich jak „choroba”, „lekarz” lub „rehabilitacja” nie przesądza o etykiecie `medical_doc`.

Anotatorzy pracują niezależnie.

Spory rozstrzyga trzeci anotator na podstawie tych wytycznych i całej treści dokumentu.
