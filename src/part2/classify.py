from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.annotation_common import LABELS


SEED = 42
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


@dataclass
class Representation:
    name: str
    family: str  # tfidf / static / encoder
    text_input: str  # formy / lematy
    data: object  # list[str] for TF-IDF (fitted inside each fold), np.ndarray otherwise
    analyzer: str = "word"


def make_model(representation: Representation) -> GridSearchCV:
    """Logistic regression with C chosen by an inner stratified 3-fold CV (macro-F1)."""
    classifier = LogisticRegression(class_weight="balanced", max_iter=5000)
    if representation.family == "tfidf":
        if representation.analyzer == "char_wb":
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=30000, sublinear_tf=True
            )
        else:
            vectorizer = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=2, sublinear_tf=True)
        pipeline = Pipeline([("features", vectorizer), ("classifier", classifier)])
    else:
        pipeline = Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
    return GridSearchCV(
        pipeline,
        {"classifier__C": C_GRID},
        scoring="f1_macro",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )


def _subset(data, indices):
    return [data[i] for i in indices] if isinstance(data, list) else data[indices]


def _fit_predict(representation: Representation, y: np.ndarray, train, test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.simplefilter("ignore", UserWarning)
        model = clone(make_model(representation))
        model.fit(_subset(representation.data, train), y[train])
        probabilities = model.predict_proba(_subset(representation.data, test))
    classes = list(model.classes_)
    full = np.zeros((len(test), len(LABELS)))
    for column, label in enumerate(classes):
        full[:, LABELS.index(label)] = probabilities[:, column]
    return full, model.best_params_["classifier__C"]


def evaluate_stratified(representation: Representation, y: np.ndarray, n_splits=5, n_repeats=5) -> dict:
    """Repeated stratified k-fold: mean/std of the fold scores and out-of-fold probabilities."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    accuracy, macro_f1, chosen_c = [], [], []
    probabilities = np.zeros((n_repeats, len(y), len(LABELS)))
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
        fold_probabilities, c = _fit_predict(representation, y, train, test)
        probabilities[fold // n_splits, test] = fold_probabilities
        predictions = np.asarray(LABELS)[fold_probabilities.argmax(axis=1)]
        accuracy.append(accuracy_score(y[test], predictions))
        macro_f1.append(f1_score(y[test], predictions, average="macro", labels=list(LABELS), zero_division=0))
        chosen_c.append(c)
    return {
        "accuracy": np.mean(accuracy),
        "accuracy_std": np.std(accuracy),
        "macro_f1": np.mean(macro_f1),
        "macro_f1_std": np.std(macro_f1),
        "probabilities": probabilities,
        "median_C": float(np.median(chosen_c)),
    }


def evaluate_leave_template_out(representation: Representation, y: np.ndarray, groups: np.ndarray) -> dict:
    """Every form template is held out once, so the test documents come from an unseen template."""
    probabilities = np.zeros((len(y), len(LABELS)))
    for train, test in LeaveOneGroupOut().split(np.zeros(len(y)), y, groups):
        probabilities[test], _ = _fit_predict(representation, y, train, test)
    predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro", labels=list(LABELS), zero_division=0),
        # "other" comes from a single template, so it can never be learned once that template is held out.
        "macro_f1_medical_official": f1_score(
            y, predictions, average="macro", labels=["medical_doc", "official_doc"], zero_division=0
        ),
        "probabilities": probabilities,
    }


def predictions_from(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(LABELS)[probabilities.argmax(axis=-1)]


def confusion(y: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        confusion_matrix(y, predictions, labels=list(LABELS)),
        index=pd.Index(LABELS, name="prawdziwa"),
        columns=pd.Index(LABELS, name="przewidziana"),
    )


# Hand-written rules taken directly from the annotation guidelines (v2). No training involved.
GUIDELINE_RULES = [
    (r"stanowisk\w*\s+komputerow", "other"),
    (r"ankieta\s+satysfakcji", "medical_doc"),
    (r"\bOL-9A?\b", "medical_doc"),
    (r"opinia\s+(lekarska|specjalist)", "medical_doc"),
    (r"skali barthel|karta oceny stanu pacjenta", "medical_doc"),
    (r"formularz oceny funkcjonowania", "medical_doc"),
    (r"ocena kliniczna", "medical_doc"),
    (r"rehabilitacji leczniczej", "medical_doc"),
]


def rule_based_label(text: str, header_characters: int = 800) -> str:
    header = text[:header_characters]
    for pattern, label in GUIDELINE_RULES:
        if re.search(pattern, header, flags=re.IGNORECASE):
            return label
    return "official_doc"
