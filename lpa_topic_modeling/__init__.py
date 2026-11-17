"""Lp(a) Reddit topic-modeling pipeline.

bge embeddings -> UMAP -> HDBSCAN -> c-TF-IDF topics, grouped by balanced k-means and
labeled with Llama-3. All stochastic steps are seeded.
"""
from . import config
from .topic_modeler import TopicModeler
from .grouping import TopicClusterer, GroupLabeler
from .figures import fig1a_temporal, fig1b_clusters, fig_silhouette_scan

__all__ = ["config", "TopicModeler", "TopicClusterer", "GroupLabeler",
           "fig1a_temporal", "fig1b_clusters", "fig_silhouette_scan"]
