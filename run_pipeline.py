#!/usr/bin/env python3
"""End-to-end Lp(a) topic-modeling pipeline (original submission).

Usage:
    python run_pipeline.py filtered_df.json --out out/

Steps: load curated discussions -> fit topic model -> save state -> group + label -> save
summaries -> render Figures 1A and 1B. Use --no-llm to fit topics without LLM labels.
"""
import argparse
import os

import pandas as pd

from lpa_topic_modeling import TopicModeler, GroupLabeler, fig1a_temporal, fig1b_clusters, config


def main():
    ap = argparse.ArgumentParser(description="Run the Lp(a) topic-modeling pipeline.")
    ap.add_argument("data", help="Curated discussions (JSON readable by pandas.read_json)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--no-llm", action="store_true", help="Skip Llama-3 labeling")
    ap.add_argument("--reduce-outliers", action="store_true",
                    help="Reassign outlier (-1) documents to existing topics")
    ap.add_argument("--outlier-strategy", default=config.OUTLIER_STRATEGY)
    ap.add_argument("--outlier-threshold", type=float, default=config.OUTLIER_THRESHOLD)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_json(args.data)
    print(f"Loaded {len(df)} discussions; seed={args.seed}")

    modeler = TopicModeler(enable_llm=not args.no_llm, seed=args.seed)
    modeler.fit_transform(df)
    n_topics = len([t for t in set(modeler.topics) if t != -1])
    print(f"Topics: {n_topics} (+ outlier)")

    if args.reduce_outliers:
        # relabel when the LLM is enabled so labels reflect post-reduction assignments
        modeler.reduce_outliers(strategy=args.outlier_strategy,
                                threshold=args.outlier_threshold, relabel=not args.no_llm)

    modeler.save_complete_state(os.path.join(args.out, "metadata.pkl"),
                                os.path.join(args.out, "model.bertopic"))

    labeler = GroupLabeler(modeler, n_groups=config.N_GROUPS, seed=args.seed,
                           use_llm_labels=not args.no_llm)
    labeler.save_group_summaries(os.path.join(args.out, "group_summaries.json"))
    for g, lab in labeler.current_labels().items():
        print(f"  {g}: {lab}")

    fig1a_temporal(df, out_path=os.path.join(args.out, "fig1a_temporal.png"))
    if modeler.embedding_model is not None:
        fig1b_clusters(modeler, labeler, seed=args.seed,
                       out_path=os.path.join(args.out, "fig1b_clusters.png"))
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
