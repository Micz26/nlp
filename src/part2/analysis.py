from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from src.part2.data import FIGURES_DIR


LABEL_COLORS = {"medical_doc": "#d1495b", "official_doc": "#00798c", "other": "#edae49"}


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    unit = normalize(vectors)
    return unit @ unit.T


def knn_label_agreement(vectors: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int = 5) -> dict:
    """Leave-one-out k-NN label accuracy; the cross-template variant ignores neighbours
    from the document's own template, so it measures meaning rather than shared layout."""
    similarity = cosine_similarity_matrix(vectors)
    np.fill_diagonal(similarity, -np.inf)
    cross = similarity.copy()
    cross[groups[:, None] == groups[None, :]] = -np.inf

    def vote(matrix):
        neighbours = np.argsort(-matrix, axis=1)[:, :k]
        predicted = [pd.Series(labels[row]).mode().iloc[0] for row in neighbours]
        return float(np.mean(np.asarray(predicted) == labels))

    return {"knn_acc": vote(similarity), "knn_acc_cross_template": vote(cross)}


def separation_metrics(vectors: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict:
    return {
        "silhouette_label": float(silhouette_score(vectors, labels, metric="cosine")),
        "silhouette_template": float(silhouette_score(vectors, groups, metric="cosine")),
        **knn_label_agreement(vectors, labels, groups),
    }


def cross_label_neighbours(vectors: np.ndarray, corpus: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Most similar document pairs whose annotated labels differ."""
    similarity = cosine_similarity_matrix(vectors)
    labels = corpus["label"].to_numpy()
    mask = np.triu(labels[:, None] != labels[None, :], k=1)
    rows, columns = np.where(mask)
    order = np.argsort(-similarity[rows, columns])
    seen, records = set(), []
    for index in order:
        left, right = rows[index], columns[index]
        key = tuple(sorted((corpus["template_id"].iat[left], corpus["template_id"].iat[right])))
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "cosine": round(float(similarity[left, right]), 3),
                "doc_a": corpus["template"].iat[left],
                "label_a": labels[left],
                "doc_b": corpus["template"].iat[right],
                "label_b": labels[right],
            }
        )
        if len(records) == top:
            break
    return pd.DataFrame(records)


def same_label_far_pairs(vectors: np.ndarray, corpus: pd.DataFrame, label: str, top: int = 5) -> pd.DataFrame:
    """Least similar template pairs inside one class (template centroids)."""
    unit = normalize(vectors)
    subset = corpus[corpus["label"] == label]
    centroids = {
        template: normalize(unit[group.index].mean(axis=0, keepdims=True))[0]
        for template, group in subset.groupby("template")
    }
    names = list(centroids)
    records = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            records.append(
                {"cosine": round(float(centroids[left] @ centroids[right]), 3), "template_a": left, "template_b": right}
            )
    return pd.DataFrame(records).sort_values("cosine").head(top)


def length_effects(vectors: np.ndarray, corpus: pd.DataFrame, raw_norms: np.ndarray | None = None) -> dict:
    """How document length relates to the vector: its raw norm (for averaged static vectors)
    and its cosine to the class centroid (how 'typical' / distinctive the vector is)."""
    unit = normalize(vectors)
    labels = corpus["label"].to_numpy()
    centroid_similarity = np.zeros(len(unit))
    for label in np.unique(labels):
        members = labels == label
        centroid = normalize(unit[members].mean(axis=0, keepdims=True))[0]
        centroid_similarity[members] = unit[members] @ centroid
    length = corpus["word_count"].to_numpy()
    result = {
        "spearman_length_vs_centroid_cos": float(spearmanr(length, centroid_similarity).statistic),
    }
    if raw_norms is not None:
        result["spearman_length_vs_norm"] = float(spearmanr(length, raw_norms).statistic)
    return result


def project(vectors: np.ndarray, method: str, seed: int = 42) -> np.ndarray:
    unit = normalize(vectors)
    if method == "umap":
        import umap

        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(unit)
    from sklearn.manifold import TSNE

    return TSNE(n_components=2, perplexity=30, metric="cosine", init="pca", random_state=seed).fit_transform(unit)


def interactive_scatter(points: np.ndarray, corpus: pd.DataFrame, title: str, file_stem: str | None = None):
    import plotly.express as px

    frame = corpus.assign(x=points[:, 0], y=points[:, 1])
    frame["fragment"] = frame["snippet"].str.slice(0, 220).str.wrap(60).str.replace("\n", "<br>")
    frame["spór w anotacji"] = frame["had_disagreement"].map({True: "tak", False: "nie"})
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="label",
        symbol="spór w anotacji",
        symbol_map={"nie": "circle", "tak": "x"},
        color_discrete_map=LABEL_COLORS,
        hover_name="template",
        hover_data={"label": True, "filename": True, "word_count": True, "fragment": True, "x": False, "y": False},
        title=title,
        width=900,
        height=650,
    )
    figure.update_traces(marker={"size": 7, "opacity": 0.8})
    figure.update_layout(legend_title_text="", xaxis_title="", yaxis_title="")
    if file_stem:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        figure.write_html(FIGURES_DIR / f"{file_stem}.html", include_plotlyjs="cdn")
    return figure


def cluster_agreement(vectors: np.ndarray, corpus: pd.DataFrame, min_cluster_size: int = 5) -> dict:
    """Unsupervised HDBSCAN clusters compared with labels and templates."""
    from sklearn.cluster import HDBSCAN

    # Euclidean distance on L2-normalized vectors is monotonic in cosine distance.
    clusters = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(normalize(vectors))
    return {
        "clusters": int(len(set(clusters)) - (1 if -1 in clusters else 0)),
        "noise_share": float(np.mean(clusters == -1)),
        "ari_vs_label": float(adjusted_rand_score(corpus["label"], clusters)),
        "ari_vs_template": float(adjusted_rand_score(corpus["template_id"], clusters)),
    }
