"""Group topics into thematic groups and label the groups with Llama-3."""
import json
import os
import re

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from umap import UMAP

from . import config


class TopicClusterer:
    """Group topics by clustering their c-TF-IDF vectors with balanced k-means.

    c-TF-IDF -> L2 normalize -> UMAP (3D, cosine, seeded) -> balanced k-means (k = N_GROUPS).
    """

    def __init__(self, topic_modeler, n_groups=config.N_GROUPS, seed=config.SEED, auto_group=True):
        self.modeler = topic_modeler
        self.n_groups = n_groups
        self.seed = seed
        self.topic_to_group = self.group_topics(n_groups) if auto_group else None

    def _topic_embedding(self):
        """L2-normalized topic c-TF-IDF vectors reduced with seeded cosine UMAP."""
        ctfidf = self.modeler.topic_model.c_tf_idf_.toarray()
        info = self.modeler.topic_model.get_topic_info()
        X = normalize(ctfidf, norm="l2", axis=1)
        emb = UMAP(
            n_components=config.GROUP_UMAP_COMPONENTS,
            n_neighbors=min(config.GROUP_UMAP_NEIGHBORS, len(X) - 1),
            min_dist=config.GROUP_UMAP_MIN_DIST, metric=config.GROUP_UMAP_METRIC,
            random_state=self.seed,
        ).fit_transform(X)
        return info, emb

    def group_topics(self, n_groups=config.N_GROUPS):
        """Return a {topic: "Group_<i>"} mapping from balanced k-means."""
        info, emb = self._topic_embedding()
        labels = self._balanced_kmeans(emb, n_groups)
        return {row["Topic"]: f"Group_{labels[i]}" for i, (_, row) in enumerate(info.iterrows())}

    def silhouette_scan(self, k_min=2, k_max=20, method="kmeans"):
        """Silhouette coefficient across candidate group counts.

        Uses the same embedding as the grouping step. method="kmeans" gives the conventional
        model-selection curve; method="balanced" uses the grouping algorithm.

        Returns:
            list of (k, score).
        """
        _, emb = self._topic_embedding()
        scores = []
        for k in range(k_min, min(k_max, len(emb) - 1) + 1):
            labels = (self._balanced_kmeans(emb, k) if method == "balanced"
                      else KMeans(n_clusters=k, random_state=self.seed, n_init=10).fit_predict(emb))
            if len(set(labels)) > 1:
                scores.append((k, float(silhouette_score(emb, labels))))
        if scores:
            best = max(scores, key=lambda t: t[1])
            print(f"Best silhouette: k={best[0]} ({best[1]:.3f})")
        return scores

    def _balanced_kmeans(self, X, k, max_iter=100):
        """k-means, then move the farthest points from oversized to undersized clusters."""
        n = X.shape[0]
        target = n // k
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        labels = km.fit_predict(X)
        dist = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        for _ in range(max_iter):
            sizes = np.bincount(labels, minlength=k)
            over = np.where(sizes > target + 1)[0]
            under = np.where(sizes < target)[0]
            if len(over) == 0 or len(under) == 0:
                break
            moved = False
            for oc in over:
                pts = np.where(labels == oc)[0]
                for p in pts[np.argsort(dist[pts, oc])[::-1]]:
                    if sizes[oc] <= target + 1:
                        break
                    cand = [u for u in under if sizes[u] < target]
                    if not cand:
                        continue
                    best = min(cand, key=lambda u: dist[p, u])
                    labels[p] = best
                    sizes[oc] -= 1
                    sizes[best] += 1
                    moved = True
            if not moved:
                break
        return labels


class GroupLabeler:
    """Build per-group summaries and generate distinct group labels with Llama-3."""

    def __init__(self, topic_modeler, clusterer=None, n_groups=config.N_GROUPS, seed=config.SEED,
                 temperature=config.LLM_TEMPERATURE, use_llm_labels=True, diversify=True,
                 preloaded_state=None):
        self.modeler = topic_modeler
        self.n_groups = n_groups
        self.seed = seed
        self.temperature = temperature
        self.diversify = diversify
        self.prompts = {}

        if preloaded_state is not None:
            self.load_group_summaries(preloaded_state)
            return

        self.clusterer = clusterer or TopicClusterer(topic_modeler, n_groups=n_groups, seed=seed)
        self.topic_to_group = self.clusterer.topic_to_group
        self.group_summaries = self._build_group_summaries()
        if use_llm_labels:
            self.generate_llm_group_labels()
            if diversify and len(self.group_summaries) > 1:
                self.diversify_llm_group_labels()

    def _topic_label(self, row):
        """Return the cleaned Llama-3 topic label, falling back to the keyword name."""
        val = row.get("LlamaCPP")
        if isinstance(val, list) and val:
            val = val[0]
        m = re.search(r"topic:\s*(.+)", str(val), re.IGNORECASE)
        return (m.group(1) if m else str(row["Name"])).strip()

    def _build_group_summaries(self):
        info = self.modeler.topic_model.get_topic_info()
        summaries = {}
        for _, row in info.iterrows():
            g = self.topic_to_group.get(row["Topic"])
            if g is None:
                continue
            s = summaries.setdefault(g, {"topics": [], "topic_labels": [], "total_count": 0,
                                         "top_words": set(), "llm_group_label": None})
            s["topics"].append(row["Topic"])
            s["topic_labels"].append(self._topic_label(row))
            s["total_count"] += int(row["Count"])
            s["top_words"].update(row["Name"].split("_")[1:][:5])
        for s in summaries.values():
            s["top_words"] = sorted(s["top_words"])[:20]
        return summaries

    # labeling
    def _llm(self):
        if getattr(self.modeler, "llm", None) is None:
            raise RuntimeError("No LLM available on the topic modeler (enable_llm=True).")
        return self.modeler.llm

    def generate_llm_group_labels(self):
        llm = self._llm()
        for g, info in self.group_summaries.items():
            prompt = (
                "You are a scientific assistant helping a cardiologist name a group of related "
                "discussion topics. Topic labels:\n- " + "\n- ".join(info["topic_labels"]) +
                "\nKeywords: " + ", ".join(info["top_words"]) +
                "\nReturn one concise, technical label in the exact format 'Group Label: <label>'."
            )
            self.prompts[g] = prompt
            out = llm(prompt, temperature=self.temperature, max_tokens=100)["choices"][0]["text"]
            m = re.search(r"(?<=Group Label:)(.*)", out)
            info["llm_group_label"] = self._clean(m.group(1) if m else out)

    def diversify_llm_group_labels(self):
        llm = self._llm()
        current = "\n".join(f"{g}: {i['llm_group_label']}" for g, i in self.group_summaries.items())
        prompt = (
            f"These {len(self.group_summaries)} health-topic groups need DISTINCT, specific labels.\n"
            f"{current}\nRewrite them as clear, mutually distinct medical labels. Respond only as "
            f"lines of the form 'Group_<n>: <label>'."
        )
        out = llm(prompt, temperature=self.temperature, max_tokens=300)["choices"][0]["text"]
        for line in out.splitlines():
            if ":" in line and "Group_" in line:
                key, lab = line.split(":", 1)
                key, lab = key.strip(), self._clean(lab)
                if key in self.group_summaries and len(lab) > 2:
                    self.group_summaries[key]["original_llm_label"] = \
                        self.group_summaries[key]["llm_group_label"]
                    self.group_summaries[key]["llm_group_label"] = lab

    @staticmethod
    def _clean(text):
        text = re.sub(r'^(Group Label:|Label:|Topic:|Answer:)\s*', "", str(text), flags=re.I)
        text = text.strip().strip('"\'').strip()
        text = re.sub(r"\s+", " ", text)
        return text[:1].upper() + text[1:] if text else text

    def update_group_labels_manual(self, updates):
        """Apply manual labels, e.g. {"Group_0": "New label"}."""
        for g, lab in updates.items():
            if g in self.group_summaries:
                self.group_summaries[g]["llm_group_label"] = lab

    # outputs
    def current_labels(self):
        return {g: i.get("llm_group_label") for g, i in self.group_summaries.items()}

    def refresh_group_counts(self):
        """Recompute each group's document total from current topic counts (use after reduce_outliers)."""
        info = self.modeler.get_cached_topic_model_info()
        counts = dict(zip(info["Topic"], info["Count"]))
        for s in self.group_summaries.values():
            s["total_count"] = int(sum(counts.get(t, 0) for t in s["topics"]))
        return self.group_summaries

    def create_labels_for_visualization(self):
        """Return a per-document group label (for Figure 1B) and the grouping info."""
        g2name = {g: (i.get("llm_group_label") or g) for g, i in self.group_summaries.items()}
        labels = [g2name.get(self.topic_to_group.get(t), "Other") for t in self.modeler.topics]
        return labels, {"topic_to_group": self.topic_to_group,
                        "group_summaries": self.group_summaries, "group_to_name": g2name}

    # persistence
    def save_group_summaries(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"group_summaries": self.group_summaries,
                       "topic_to_group": self.topic_to_group}, f, indent=2, default=str)
        print(f"Saved group summaries -> {path}")

    def load_group_summaries(self, path):
        with open(path) as f:
            data = json.load(f)
        self.group_summaries = data["group_summaries"]
        self.topic_to_group = {int(k) if str(k).lstrip("-").isdigit() else k: v
                               for k, v in data.get("topic_to_group", {}).items()}
        print(f"Loaded {len(self.group_summaries)} groups from {path}")
        return self
