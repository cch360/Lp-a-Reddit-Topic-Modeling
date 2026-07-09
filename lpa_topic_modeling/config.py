
import torch

# --- reproducibility ---
SEED = 42
LLM_TEMPERATURE = 0.0          # greedy decoding for deterministic labels

# --- models ---
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_REPO = "NousResearch/Meta-Llama-3-8B-Instruct-GGUF"
LLM_FILE = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
LLM_CTX = 8192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- preprocessing ---
TEXT_COLUMN = "body"
MIN_DOC_LEN = 25               # characters, after cleaning
ENGLISH_ONLY = True

# --- topic model (document level) ---
MIN_CLUSTER_SIZE = 100
MIN_SAMPLES = 10
N_NEIGHBORS = 10
MIN_DIST = 0.0
N_COMPONENTS = 2               # UMAP dims before HDBSCAN (as run)
UMAP_METRIC = "euclidean"
NGRAM_RANGE = (1, 3)
MIN_TOPIC_SIZE = 10            # topics below this fold into the outlier topic

# --- optional outlier reduction (reassign topic -1 docs to existing topics) ---
OUTLIER_STRATEGY = "c-tf-idf"  # "c-tf-idf" (fast) or "embeddings" (re-embeds, more faithful)
OUTLIER_THRESHOLD = 0.1        # lower -> reassign more (fewer remaining outliers)

# --- grouping (topic level) ---
N_GROUPS = 10                  # set a priori
GROUP_UMAP_COMPONENTS = 3
GROUP_UMAP_NEIGHBORS = 15
GROUP_UMAP_MIN_DIST = 0.1
GROUP_UMAP_METRIC = "cosine"

# --- topic-label prompt (Llama-3) ---
TOPIC_PROMPT = (
    "You are an honest, scientific chatbot that helps me, a cardiologist, create a single "
    "representative label for a topic. The topic is described by the following keywords: "
    "[KEYWORDS]. Representative documents from the topic: [DOCUMENTS]. Based on the above, give "
    "a short label in the exact format 'topic: <description>'. Do not add any other commentary."
)

# --- Fig 1A milestone overlay ---
MILESTONES = [
    (2018, "2018 AHA/ACC cholesterol guideline", "guideline"),
    (2019, "2019 ESC/EAS dyslipidaemia guideline", "guideline"),
    (2022, "2022 EAS consensus statement", "guideline"),
    (2024, "2024 NLA focused statement", "guideline"),
]
