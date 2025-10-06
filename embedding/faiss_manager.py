import numpy as np
import pandas as pd
import faiss

from config.settings import INDEX_FILE
from utils.logging_config import setup_logging

log = setup_logging()

def build_faiss(df: pd.DataFrame):
    embeddings = np.vstack(df["embedding"].to_numpy())
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    log.info("FAISS index saved to %s", INDEX_FILE)
    return index