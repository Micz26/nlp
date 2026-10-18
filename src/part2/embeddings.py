from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.part2.data import CACHE_DIR


EMBEDDINGS_DIR = CACHE_DIR / "embeddings"


@dataclass(frozen=True)
class StaticModel:
    name: str
    repo: str
    filename: str
    kind: str  # "word2vec" (gensim KeyedVectors) or "fasttext" (Facebook binary with subwords)
    trained_on: str


@dataclass(frozen=True)
class Encoder:
    name: str
    repo: str
    prefix: str
    window_tokens: int
    long_context_tokens: int | None = None


STATIC_MODELS = {
    "w2v": StaticModel(
        "w2v", "clarin-pl/word2vec-kgr10", "cbow.v300.m8.hs.mwe.w2v.gensim", "word2vec", "lematy"
    ),
    "fasttext": StaticModel(
        "fasttext", "clarin-pl/fastText-kgr10", "kgr10.plain.skipgram.dim100.neg10.bin", "fasttext", "formy"
    ),
}

ENCODERS = {
    # The MMLW card requires the "zapytanie: " prefix; symmetric tasks put it on every text.
    "mmlw": Encoder("mmlw", "sdadas/mmlw-roberta-base", "zapytanie: ", window_tokens=500),
    "bge-m3": Encoder("bge-m3", "BAAI/bge-m3", "", window_tokens=500, long_context_tokens=8192),
}


def _cached(name: str, compute) -> np.ndarray:
    path = EMBEDDINGS_DIR / f"{name}.npy"
    if path.exists():
        return np.load(path)
    vectors = compute()
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def load_static_model(model: StaticModel):
    from gensim.models import KeyedVectors
    from gensim.models.fasttext import load_facebook_vectors
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model.repo, model.filename)
    if model.kind == "word2vec":
        hf_hub_download(model.repo, model.filename + ".vectors.npy")
        return KeyedVectors.load(path, mmap="r")
    return load_facebook_vectors(path)


def _lookup(keyed_vectors, token: str, use_subwords: bool) -> np.ndarray | None:
    # KGR10 keeps acronyms and proper names in their original case (e.g. "ZUS").
    for candidate in (token, token.upper(), token.capitalize()):
        if candidate in keyed_vectors.key_to_index:
            return keyed_vectors[candidate]
    if use_subwords:
        return keyed_vectors[token]
    return None


def static_document_vectors(
    keyed_vectors, documents: list[list[str]], use_subwords: bool, weighting: str = "mean"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Averages word vectors per document.

    weighting="mean" is the plain average; weighting="sif" uses the smooth inverse frequency
    weights a / (a + p(w)) and removes the first principal component (Arora et al., 2017).
    Returns the vectors, the in-vocabulary rate per document and the raw (unnormalized) norms.
    """
    frequencies = Counter(token for document in documents for token in document)
    total = sum(frequencies.values())
    smoothing = 1e-3
    vectors, coverage = [], []
    cache: dict[str, np.ndarray | None] = {}
    in_vocabulary: dict[str, bool] = {}
    for document in documents:
        rows, weights, known = [], [], 0
        for token in document:
            if token not in cache:
                cache[token] = _lookup(keyed_vectors, token, use_subwords)
                in_vocabulary[token] = any(
                    candidate in keyed_vectors.key_to_index
                    for candidate in (token, token.upper(), token.capitalize())
                )
            vector = cache[token]
            known += in_vocabulary[token]
            if vector is None:
                continue
            rows.append(vector)
            weights.append(
                smoothing / (smoothing + frequencies[token] / total) if weighting == "sif" else 1.0
            )
        coverage.append(known / max(len(document), 1))
        if rows:
            vectors.append(np.average(np.asarray(rows, dtype=np.float32), axis=0, weights=weights))
        else:
            vectors.append(np.zeros(keyed_vectors.vector_size, dtype=np.float32))
    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1)
    if weighting == "sif":
        component = np.linalg.svd(matrix, full_matrices=False)[2][0]
        matrix = matrix - np.outer(matrix @ component, component)
    return matrix, np.asarray(coverage), norms


def compute_static_embeddings(tokens: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Static embeddings for every model x input (forms / lemmas) x weighting combination."""
    results: dict[str, dict[str, np.ndarray]] = {}
    for model in STATIC_MODELS.values():
        keyed_vectors = None
        for column in ("forms", "lemmas"):
            for weighting in ("mean", "sif"):
                key = f"{model.name}_{column}_{weighting}"
                paths = [EMBEDDINGS_DIR / f"{key}{suffix}.npy" for suffix in ("", "_coverage", "_norm")]
                if not all(path.exists() for path in paths):
                    if keyed_vectors is None:
                        keyed_vectors = load_static_model(model)
                    matrix, coverage, norms = static_document_vectors(
                        keyed_vectors,
                        [list(document) for document in tokens[column]],
                        use_subwords=model.kind == "fasttext",
                        weighting=weighting,
                    )
                    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
                    for path, value in zip(paths, (matrix, coverage, norms)):
                        np.save(path, value)
                results[key] = {
                    name: np.load(path) for name, path in zip(("vectors", "coverage", "norm"), paths)
                }
        del keyed_vectors
    return results


def split_into_windows(tokenizer, text: str, window_tokens: int) -> list[str]:
    """Cuts the text into consecutive pieces of at most window_tokens subword tokens."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    offsets = encoding["offset_mapping"]
    windows = []
    for start in range(0, len(offsets), window_tokens):
        piece = offsets[start : start + window_tokens]
        windows.append(text[piece[0][0] : piece[-1][1]])
    return windows or [text]


def load_encoder(encoder: Encoder):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(encoder.repo, device=device)


def encode_documents(
    encoder: Encoder, texts: list[str], suffix: str = "", model=None
) -> dict[str, np.ndarray]:
    """Three ways of fitting long documents into an encoder:

    head   - only the first window (what plain truncation gives),
    chunks - mean of L2-normalized window embeddings over the whole document,
    long   - one pass over up to long_context_tokens tokens (only for long-context models).
    """
    names = [f"{encoder.name}{suffix}_{strategy}" for strategy in ("head", "chunks")]
    if encoder.long_context_tokens:
        names.append(f"{encoder.name}{suffix}_long")
    if all((EMBEDDINGS_DIR / f"{name}.npy").exists() for name in names):
        return {name.rsplit("_", 1)[1]: np.load(EMBEDDINGS_DIR / f"{name}.npy") for name in names}

    model = model or load_encoder(encoder)
    model.max_seq_length = encoder.window_tokens + 16
    windows = [split_into_windows(model.tokenizer, text, encoder.window_tokens) for text in texts]
    flat = [encoder.prefix + window for document in windows for window in document]
    flat_vectors = model.encode(
        flat, batch_size=16, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    )
    head, chunks, position = [], [], 0
    for document in windows:
        block = flat_vectors[position : position + len(document)]
        position += len(document)
        head.append(block[0])
        mean = block.mean(axis=0)
        chunks.append(mean / np.linalg.norm(mean))
    results = {"head": np.vstack(head), "chunks": np.vstack(chunks)}

    if encoder.long_context_tokens:
        model.max_seq_length = encoder.long_context_tokens
        results["long"] = model.encode(
            [encoder.prefix + text for text in texts],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    for strategy, vectors in results.items():
        np.save(EMBEDDINGS_DIR / f"{encoder.name}{suffix}_{strategy}.npy", vectors)
    np.save(
        EMBEDDINGS_DIR / f"{encoder.name}{suffix}_window_counts.npy",
        np.asarray([len(document) for document in windows]),
    )
    return results


def compute_encoder_embeddings(corpus: pd.DataFrame, lemmas: pd.DataFrame) -> dict[str, np.ndarray]:
    results = {}
    for encoder in ENCODERS.values():
        for strategy, vectors in encode_documents(encoder, corpus["clean_text"].tolist()).items():
            results[f"{encoder.name}_{strategy}"] = vectors
        if encoder.name == "mmlw":
            # Control experiment: what happens when a contextual encoder gets lemmatized input.
            lemma_texts = [" ".join(document) for document in lemmas["lemmas"]]
            for strategy, vectors in encode_documents(encoder, lemma_texts, suffix="-lemmas").items():
                results[f"{encoder.name}-lemmas_{strategy}"] = vectors
    return results


def encode_texts(encoder_name: str, texts: list[str], model=None) -> np.ndarray:
    encoder = ENCODERS[encoder_name]
    model = model or load_encoder(encoder)
    return model.encode(
        [encoder.prefix + text for text in texts], normalize_embeddings=True, convert_to_numpy=True
    )
