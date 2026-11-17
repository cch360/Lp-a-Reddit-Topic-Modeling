# Lp(a) Reddit Topic Modeling

Code for "Lipoprotein(a): Large Language Model-Based Social Media Topic Modeling" (J Am Heart Assoc, under review).

## Contents

- `LpA_reddit_JAHA_revision.ipynb` - analysis for the revised manuscript; reported values are written to `revision_results.json`.
- `lpa_topic_modeling/` - topic model, thematic grouping, and figures.
- `LpA_reddit_paper_driver.ipynb`, `run_pipeline.py` - original submission pipeline (original table and figure numbering).

## Methods

- **Embedding:** BAAI/bge-base-en-v1.5.
- **Topics:** UMAP + HDBSCAN (5-D, leaf selection, min cluster size 75), chosen by cross-seed stability (Table S2); outliers reassigned by c-TF-IDF (threshold 0.1).
- **Groups:** balanced k-means on topic c-TF-IDF vectors; k = 9 by mean silhouette over k = 4-12 and five seeds (Table S3).
- **Labels:** Llama-3 8B (temperature 0), reviewed by two clinicians and adjudicated (Tables S4, S11).
- **Seed:** 42 for all stochastic steps.

## Running

Revision notebook: Google Colab (A100), Python 3.13; package versions in Table S8. Set `V3_PATH` and `BASE` before running. `RESUME = True` reloads saved models; `RESUME = False` refits (~1 hour with Llama-3). An optional, untracked `name_mask.json` in `BASE` masks names of individuals.

Original pipeline:

```bash
pip install -r requirements.txt
python run_pipeline.py filtered_df.json --out out/ [--no-llm]
```

## Data availability

Reddit content, trained models, rater sheets, and the name-mask file are not included, in accordance with platform policies. Search terms and matching patterns are in the Supplemental Material. De-identified data are available from the corresponding author on reasonable request.

## License

MIT
