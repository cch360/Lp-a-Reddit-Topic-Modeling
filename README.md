# Lp(a) Reddit Topic Modeling (v3)

Pipeline behind the manuscript "Lipoprotein(a): Large Language Model-Based
Social Media Topic Modeling". Characterizes public Lp(a) discourse on Reddit
with embedding-based topic modeling and LLM labeling. Seeded, so runs are
reproducible by default.

## Pipeline

1. Embed each discussion with BAAI/bge-base-en-v1.5 (768-dim).
2. Reduce with UMAP (seeded) and cluster with HDBSCAN to get topics.
3. Describe topics with c-TF-IDF keywords and label them with Llama-3 (temperature 0).
4. Group the topics' c-TF-IDF vectors (L2-normalize, UMAP to 3D, balanced k-means, k=10).
5. Label groups with Llama-3, then render Figures 1A and 1B.

## Layout

```
lpa_topic_modeling/
  config.py         seeds, temperature, model versions, hyperparameters, milestones
  topic_modeler.py  TopicModeler: embed, UMAP, HDBSCAN, c-TF-IDF, optional Llama-3 topic labels
  grouping.py       TopicClusterer (balanced k-means) and GroupLabeler (Llama-3 group labels)
  figures.py        fig1a_temporal (with milestone overlay), fig1b_clusters
run_pipeline.py     end-to-end driver
```

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py filtered_df.json --out out/            # full run (GPU recommended for the LLM)
python run_pipeline.py filtered_df.json --out out/ --no-llm   # topics only, no labels
```

Outputs: metadata.pkl and model.bertopic (reload with TopicModeler().load_complete_state(...)),
group_summaries.json (reload with GroupLabeler(modeler, preloaded_state=...)), and the figures.

## Reproducibility

Every stochastic step is seeded from config.SEED (document-level UMAP, grouping UMAP, k-means),
and the LLM runs at temperature 0 with a fixed seed, so a rerun reproduces the same topics,
groups, and labels. The trained model, assignments, and labels are saved as artifacts; load them
rather than regenerating when you just need the published results.
