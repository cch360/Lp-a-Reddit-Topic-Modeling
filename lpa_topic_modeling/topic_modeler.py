"""Document embedding, dimensionality reduction, clustering, and LLM topic labeling."""
import os
import pickle

import numpy as np
import pandas as pd
import umap
import hdbscan
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from . import config

try:
    from bertopic.representation import LlamaCPP
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
    LLAMA_AVAILABLE = True
except ImportError:
    LlamaCPP = Llama = hf_hub_download = None
    LLAMA_AVAILABLE = False


class TopicModeler:
    """bge embeddings -> UMAP -> HDBSCAN -> c-TF-IDF, with optional Llama-3 topic labels."""

    def __init__(self, enable_llm=False, seed=config.SEED, cache_dir="models", offline_mode=False,
                 embedding_model_name=config.EMBEDDING_MODEL,
                 min_cluster_size=config.MIN_CLUSTER_SIZE, min_samples=config.MIN_SAMPLES,
                 n_neighbors=config.N_NEIGHBORS, min_dist=config.MIN_DIST,
                 n_components=config.N_COMPONENTS, ngram_range=config.NGRAM_RANGE):
        self.seed = seed
        self.embedding_model_name = embedding_model_name
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.ngram_range = ngram_range
        self.enable_llm = enable_llm and LLAMA_AVAILABLE

        self.embedding_model = None
        self.topic_model = None
        self.topics = None
        self.probs = None
        self.texts = None
        self.df = None
        self.llm = None
        self._cached_topic_model_info = None

        if not offline_mode:
            self.embedding_model = SentenceTransformer(embedding_model_name, device=config.DEVICE)

        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,
            metric=config.UMAP_METRIC, random_state=seed,
        )
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
            metric="euclidean", cluster_selection_method="eom",
        )
        self.vectorizer_model = CountVectorizer(stop_words="english", ngram_range=ngram_range)

        self.representation_model = {"KeyBERT": KeyBERTInspired()}
        if self.enable_llm and not offline_mode:
            self.llm = self._load_llm(cache_dir)
            self.representation_model["LlamaCPP"] = LlamaCPP(
                self.llm, prompt=config.TOPIC_PROMPT, tokenizer="char", doc_length=2000,
                pipeline_kwargs={"temperature": config.LLM_TEMPERATURE, "seed": seed},
            )

    def _load_llm(self, cache_dir):
        """Load the quantized Llama-3 model, downloading it to cache_dir if needed."""
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, config.LLM_FILE)
        if not os.path.exists(path):
            path = hf_hub_download(repo_id=config.LLM_REPO, filename=config.LLM_FILE, cache_dir=cache_dir)
        return Llama(model_path=path, n_gpu_layers=-1, n_ctx=config.LLM_CTX,
                     seed=self.seed, verbose=False)

    def _load_bertopic(self, model_path):
        """Load a saved BERTopic model without instantiating the pickled Llama model."""
        llama = None
        try:
            import llama_cpp as llama
            _real = llama.Llama.__init__
            llama.Llama.__init__ = (
                lambda s, *a, **k: setattr(s, "model_path", k.get("model_path", a[0] if a else None)))
        except Exception:
            llama = None
        try:
            model = BERTopic.load(model_path)
        finally:
            if llama is not None:
                llama.Llama.__init__ = _real
        model.representation_model = None
        return model

    # pipeline
    def preprocess_texts(self, df, text_column=config.TEXT_COLUMN,
                         min_length=config.MIN_DOC_LEN, filter_english_only=config.ENGLISH_ONLY):
        """Clean text, drop short documents, optionally keep English only; return the texts."""
        df = df.copy()
        df[text_column] = (df[text_column].fillna("")
                           .str.replace(r"http\S+", "", regex=True)
                           .str.replace(r"\\n", " ", regex=True)
                           .str.replace("&gt;", "", regex=False)
                           .str.replace("&lt;", "", regex=False)
                           .str.replace("&amp;", "&", regex=False)
                           .str.replace(r"\s+", " ", regex=True).str.strip())
        df["body_len"] = df[text_column].str.len()
        df = df[df["body_len"] >= min_length]

        if filter_english_only:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0

            def is_en(t):
                try:
                    return detect(t) == "en" if len(t.strip()) >= 10 else True
                except Exception:
                    return True
            df["is_english"] = df[text_column].apply(is_en)
            df = df[df["is_english"]]

        self.df = df.reset_index(drop=True)
        return self.df[text_column].tolist()

    def fit_transform(self, df, text_column=config.TEXT_COLUMN, preprocess=True,
                      min_topic_size=config.MIN_TOPIC_SIZE):
        """Fit BERTopic on the (optionally preprocessed) texts and return topic assignments."""
        texts = self.preprocess_texts(df, text_column) if preprocess else df[text_column].tolist()
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model, umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model, vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model, verbose=True,
        )
        self.texts = texts
        self.topics, self.probs = self.topic_model.fit_transform(texts)
        if min_topic_size > 0:
            self.filter_small_topics(min_topic_size)
        return self.topics

    def filter_small_topics(self, min_topic_size=config.MIN_TOPIC_SIZE):
        """Move documents in topics smaller than min_topic_size to the outlier topic (-1)."""
        info = self.topic_model.get_topic_info()
        small = info[info["Count"] < min_topic_size]["Topic"].tolist()
        if not small:
            return
        self.topics = [(-1 if t in small else t) for t in self.topics]
        self.topic_model.topics_ = np.array(self.topics)

    def reduce_outliers(self, strategy=config.OUTLIER_STRATEGY,
                        threshold=config.OUTLIER_THRESHOLD, relabel=False):
        """Reassign outlier (-1) documents to existing topics and refresh topic representations."""
        if self.topic_model is None:
            raise RuntimeError("Fit or load a model first.")
        before = sum(t == -1 for t in self.topics)
        self.topics = self.topic_model.reduce_outliers(
            self.texts, self.topics, strategy=strategy, threshold=threshold)
        rep = self.representation_model if relabel else None  # c-TF-IDF only; no embeddings/LLM
        self.topic_model.update_topics(
            self.texts, topics=self.topics,
            vectorizer_model=CountVectorizer(stop_words="english", ngram_range=self.ngram_range),
            representation_model=rep)
        after = sum(t == -1 for t in self.topics)
        print(f"Outliers reduced: {before} -> {after} "
              f"({100 * after / len(self.topics):.1f}% of {len(self.topics)}) via '{strategy}'")
        return self.topics

    def drop_and_refit(self, allowed_topics):
        """Keep documents in allowed_topics and refit the model from scratch."""
        mask = [t in allowed_topics for t in self.topics]
        self.df = self.df[mask].reset_index(drop=True)
        # fresh (seeded) components for the refit
        self.__init__(enable_llm=self.enable_llm, seed=self.seed,
                      embedding_model_name=self.embedding_model_name,
                      min_cluster_size=self.min_cluster_size, min_samples=self.min_samples,
                      n_neighbors=self.n_neighbors, min_dist=self.min_dist,
                      n_components=self.n_components, ngram_range=self.ngram_range,
                      offline_mode=True)
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=config.DEVICE)
        return self.fit_transform(self.df, preprocess=False)

    # access
    def topic_model_info(self):
        return self.topic_model.get_topic_info()

    def get_cached_topic_model_info(self):
        if self.topic_model is not None:
            return self.topic_model_info()
        return self._cached_topic_model_info

    # persistence
    def save_complete_state(self, metadata_path, model_path):
        if self.topic_model is None:
            raise RuntimeError("Nothing to save; fit a model first.")
        for p in (metadata_path, model_path):
            os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
        metadata = {
            "topics": self.topics, "probs": self.probs, "texts": self.texts, "df": self.df,
            "topic_model_info": self.topic_model_info(),
            "model_parameters": {
                "embedding_model_name": self.embedding_model_name, "seed": self.seed,
                "min_cluster_size": self.min_cluster_size, "min_samples": self.min_samples,
                "n_neighbors": self.n_neighbors, "min_dist": self.min_dist,
                "n_components": self.n_components, "ngram_range": self.ngram_range,
            },
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        self.topic_model.save(model_path, save_ctfidf=True, save_embedding_model=False)
        print(f"Saved metadata -> {metadata_path}; model -> {model_path}")

    def load_complete_state(self, metadata_path, model_path=None):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        self.topics = metadata.get("topics")
        self.probs = metadata.get("probs")
        self.texts = metadata.get("texts")
        self.df = metadata.get("df")
        self._cached_topic_model_info = metadata.get("topic_model_info")
        for k, v in (metadata.get("model_parameters") or {}).items():
            if v is not None:
                setattr(self, k, v)
        if model_path:
            self.topic_model = self._load_bertopic(model_path)
            if (getattr(self.topic_model, "embedding_model", None) is None
                    and self.embedding_model is not None):
                self.topic_model.embedding_model = self.embedding_model
        print(f"Loaded {len(self.texts or [])} texts; "
              f"{len(set(self.topics or [])) } unique topics")
        return self
