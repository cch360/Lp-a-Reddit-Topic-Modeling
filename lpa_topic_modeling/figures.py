"""Manuscript figures: 1A (temporal volume with milestones), 1B (UMAP topic groups), and the silhouette scan."""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import umap

from . import config

# styling
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
})

_MS_COLORS = {"guideline": "#2CA02C", "trial": "#D62728", "both": "#7F3F9B"}
_COMMENT, _POST = "#6BAED6", "#08519C"
_TOL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
        "#AA3377", "#BBBBBB", "#EE8866", "#77AADD", "#999933"]


def _save(fig, out_path):
    """Save fig as PDF, EPS, PNG and TIFF using the stem of out_path."""
    if not out_path:
        return
    stem = os.path.splitext(out_path)[0]
    for ext in ("pdf", "eps", "png", "tiff"):
        try:
            fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"  ({ext} skipped: {e})")
    print(f"Saved {stem}.[pdf/eps/png/tiff]")


def fig1a_temporal(df, milestones=config.MILESTONES, created_col="created_utc",
                   type_col="type_of_entry", out_path=None, figsize=(7.5, 3.4)):
    """Annual stacked posts and comments with numbered milestone markers."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[created_col]):
        df[created_col] = pd.to_datetime(df[created_col], unit="s", utc=True)
    df["year"] = df[created_col].dt.year
    pivot = (df.groupby(["year", type_col]).size().reset_index(name="n")
             .pivot(index="year", columns=type_col, values="n").fillna(0))
    pivot.index = pivot.index.astype(str)
    order = [c for c in ("comment", "post") if c in pivot.columns] + \
            [c for c in pivot.columns if c not in ("comment", "post")]
    pivot = pivot[order]
    colmap = {"comment": _COMMENT, "post": _POST}
    colors = [colmap.get(c, "#999999") for c in order]

    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.82,
               edgecolor="white", linewidth=0.3, legend=False, zorder=2)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Number of discussions (posts + comments)", fontsize=9.5)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E9E9E9", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # numbered milestone markers
    pos = {yr: i for i, yr in enumerate(pivot.index)}
    ymax = pivot.sum(axis=1).max()
    ytop = ymax * 1.05
    for n, (year, _label, kind) in enumerate(milestones, 1):
        if str(year) not in pos:
            continue
        x = pos[str(year)]
        ax.axvline(x, color=_MS_COLORS[kind], ls=(0, (5, 3)), lw=1.0, zorder=1)
        ax.plot(x, ytop, "o", ms=13, mfc="white", mec=_MS_COLORS[kind], mew=1.3, zorder=4)
        ax.text(x, ytop, str(n), color=_MS_COLORS[kind], ha="center", va="center",
                fontsize=7.5, fontweight="bold", zorder=5)
    ax.set_ylim(0, ymax * 1.12)

    # content-type legend
    ct = [Patch(fc=colmap.get(c, "#999999"), label=c) for c in order]
    ax.legend(handles=ct, title="Content type", loc="upper left",
              fontsize=8, title_fontsize=8.5, frameon=True)
    ax.text(-0.075, 1.04, "(A)", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom")

    fig.tight_layout()
    _save(fig, out_path)
    return fig, pivot, milestones


def fig1b_clusters(modeler, labeler, seed=config.SEED, out_path=None, figsize=(8.5, 7.6),
                   n_neighbors=15, min_dist=0.01, metric="euclidean", remove_outliers=True,
                   alpha=0.6, s=7, emb2d=None):
    """UMAP projection of the document embeddings, colored by topic group."""
    labels, _ = labeler.create_labels_for_visualization()
    labels = np.array(labels)
    if emb2d is None:
        emb = modeler.embedding_model.encode(list(modeler.texts), show_progress_bar=True)
        emb2d = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist,
                          metric=metric, random_state=seed).fit_transform(emb)
    emb2d = np.asarray(emb2d)

    if remove_outliers:
        q1, q3 = np.percentile(emb2d, 25, axis=0), np.percentile(emb2d, 75, axis=0)
        iqr = q3 - q1
        keep = np.all((emb2d >= q1 - 1.5 * iqr) & (emb2d <= q3 + 1.5 * iqr), axis=1)
        emb2d, labels = emb2d[keep], labels[keep]

    uniq = sorted(set(labels))
    colors = {lab: _TOL[i % len(_TOL)] for i, lab in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=figsize)
    for lab in uniq:
        m = labels == lab
        ax.scatter(emb2d[m, 0], emb2d[m, 1], c=[colors[lab]], label=lab,
                   alpha=alpha, s=s, linewidths=0, rasterized=True)
    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.02)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(-0.03, 1.02, "(B)", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom")
    # full group names, two-column legend beneath the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2,
              fontsize=7, title="Topic groups", title_fontsize=9, frameon=False,
              markerscale=2.6, columnspacing=1.0, handletextpad=0.4, borderaxespad=0)
    fig.tight_layout()
    _save(fig, out_path)
    return fig, emb2d, labels


def fig_silhouette_scan(clusterer, k_min=2, k_max=20, method="kmeans", out_path=None, figsize=(6, 4)):
    """Silhouette coefficient vs. number of groups (k), marking config.N_GROUPS and the maximum."""
    scores = clusterer.silhouette_scan(k_min, k_max, method=method)
    ks = [k for k, _ in scores]
    vals = [s for _, s in scores]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, vals, marker="o")
    ax.axvline(config.N_GROUPS, ls="--", color="grey", lw=1, label=f"k = {config.N_GROUPS} (selected)")
    if scores:
        best = max(scores, key=lambda t: t[1])
        ax.scatter([best[0]], [best[1]], color="red", zorder=5, label=f"max (k = {best[0]})")
    ax.set_xlabel("Number of groups (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(ks)
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    return fig, scores
