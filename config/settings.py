import os
from pathlib import Path

# Paths
ROOT = Path(".")
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
EMBEDDINGS_PARQUET = DATA_DIR / "embeddings.parquet"
INDEX_FILE = DATA_DIR / "faiss.index"
METADATA_FILE = DATA_DIR / "metadata.csv"
TRAIN_QA_FILE = DATA_DIR / "train_QA.csv"
TEST_Q_FILE = DATA_DIR / "test_Q.csv"
SUBMISSION_FILE = "submission.csv"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Retrieval/rerank sizes
TOP_K = 30
TOP_K_RERANK = 5
EMBED_BATCH = 32
TFIDF_MAX_FEATURES = 8000

# Retriever / reranker / LLM defaults
RETRIEVER_BASE = "BAAI/bge-small-en-v1.5"
RETRIEVER_OUT = Path("models/fine_tuned_retriever")
RETRIEVE_EPOCHS = 2
RETRIEVE_BATCH = 16
RETRIEVE_LR = 2e-5
RETRIEVE_MAX_TRAIN_EXAMPLES = None

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# LLM generation model
HF_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
HF_MAX_NEW_TOKENS = 256
HF_TEMPERATURE = 0.7

# Chunk token limit when building chunks
CHUNK_TOKENS = 300

# Logging
LOGFILE = Path("logs/wattbot_run.log")