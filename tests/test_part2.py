import numpy as np
import pandas as pd

from src.part2.analysis import knn_label_agreement
from src.part2.classify import rule_based_label
from src.part2.data import clean_text, tokenize
from src.part2.embeddings import split_into_windows, static_document_vectors


class FakeTokenizer:
    """Whitespace tokenizer with the same call signature as a Hugging Face tokenizer."""

    def __call__(self, text, add_special_tokens, return_offsets_mapping, verbose):
        offsets, position = [], 0
        for word in text.split():
            start = text.index(word, position)
            offsets.append((start, start + len(word)))
            position = start + len(word)
        return {"offset_mapping": offsets}


class FakeKeyedVectors:
    def __init__(self, vectors):
        self.key_to_index = {word: index for index, word in enumerate(vectors)}
        self.vectors = vectors
        self.vector_size = len(next(iter(vectors.values())))

    def __getitem__(self, word):
        return np.asarray(self.vectors.get(word, np.ones(self.vector_size)), dtype=np.float32)


def test_clean_text_removes_layout_noise() -> None:
    raw = "| :--- | :--- |\n**Wniosek** <br> o zasiłek ........................ | X |"

    assert clean_text(raw) == "Wniosek o zasiłek X"


def test_tokenize_keeps_polish_words_only() -> None:
    assert tokenize("PESEL 12345, Zaświadczenie o stanie zdrowia!") == ["pesel", "zaświadczenie", "stanie", "zdrowia"]


def test_windows_cover_whole_text_in_order() -> None:
    text = "jeden dwa trzy cztery pięć"
    windows = split_into_windows(FakeTokenizer(), text, window_tokens=2)

    assert windows == ["jeden dwa", "trzy cztery", "pięć"]


def test_static_vectors_skip_unknown_words_without_subwords() -> None:
    keyed_vectors = FakeKeyedVectors({"lekarz": [1.0, 0.0], "ZUS": [0.0, 1.0]})
    vectors, coverage, _ = static_document_vectors(
        keyed_vectors, [["lekarz", "zus", "nieznane"]], use_subwords=False
    )

    np.testing.assert_allclose(vectors[0], [0.5, 0.5])
    assert coverage[0] == 2 / 3


def test_rules_follow_guidelines() -> None:
    assert rule_based_label("ZUS OL-9 Zaświadczenie o stanie zdrowia") == "medical_doc"
    assert rule_based_label("Lista kontrolna: stanowisko\nkomputerowe") == "other"
    assert rule_based_label("PIT-37 Zeznanie o wysokości osiągniętego dochodu") == "official_doc"


def test_cross_template_knn_ignores_own_template() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]])
    labels = np.array(["a", "a", "b", "b"])
    groups = np.array([0, 0, 1, 1])
    result = knn_label_agreement(vectors, labels, groups, k=1)

    assert result["knn_acc"] == 1.0
    assert result["knn_acc_cross_template"] == 0.0
