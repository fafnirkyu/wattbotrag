import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

from config.settings import EMBEDDINGS_PARQUET, TFIDF_MAX_FEATURES
from utils.logging_config import setup_logging

log = setup_logging()

class Embedder:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.all_texts_for_tfidf = None

    def embed_chunks(self, chunks: list, model_name: str, device: str = None, batch_size: int = 32) -> pd.DataFrame:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("Embedding chunks using model %s on device %s", model_name, device)
        model = SentenceTransformer(model_name, device=device)

        texts = [c["text"] for c in chunks]
        embeddings = []

        try:
            if device.startswith("cuda"):
                amp_ctx = torch.cuda.amp.autocast
                ctx = amp_ctx("cuda", dtype=torch.float16)
            else:
                class DummyCtx:
                    def __enter__(self): return None
                    def __exit__(self, *args): return False
                ctx = DummyCtx()

            with ctx:
                for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
                    batch = texts[i:i+batch_size]
                    try:
                        batch_emb = model.encode(batch, show_progress_bar=False, convert_to_tensor=False)
                    except TypeError:
                        batch_emb = model.encode(batch, show_progress_bar=False)
                    embeddings.extend(batch_emb)
        except Exception as e:
            log.warning("Batch encoding failed, falling back to single-shot encode: %s", e)
            try:
                embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=False)
            except Exception as ex:
                log.error("Embedding failed: %s", ex)
                raise

        embeddings = np.array(embeddings, dtype=np.float32)

        df = pd.DataFrame({
            "doc_id": [c["doc_id"] for c in chunks],
            "pages": [c["pages"] for c in chunks],
            "text": texts,
            "embedding": list(embeddings)
        })

        self.all_texts_for_tfidf = texts
        self.tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1,2))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_texts_for_tfidf)

        try:
            df.to_parquet(EMBEDDINGS_PARQUET, index=False)
            log.info("Saved embeddings & TF-IDF to %s", EMBEDDINGS_PARQUET)
        except Exception as e:
            log.warning("Failed to save embeddings parquet: %s", e)

        return df

    def embed_chunks_with_model(self, chunks: list, model_or_path, batch_size: int = 32) -> pd.DataFrame:
        if model_or_path is None:
            raise ValueError("model_or_path is None")
        if isinstance(model_or_path, (str, Path)):
            model = SentenceTransformer(str(model_or_path), device="cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(model_or_path, SentenceTransformer):
            model = model_or_path
        else:
            model = SentenceTransformer(str(model_or_path), device="cuda" if torch.cuda.is_available() else "cpu")

        texts = [c["text"] for c in chunks]
        embeddings = []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            if device.startswith("cuda"):
                amp_ctx = torch.cuda.amp.autocast
                ctx = amp_ctx(device_type="cuda", dtype=torch.float16)
            else:
                class DummyCtx:
                    def __enter__(self): return None
                    def __exit__(self, *args): return False
                ctx = DummyCtx()

            with ctx:
                for i in tqdm(range(0, len(texts), batch_size), desc="Re-embedding batches"):
                    batch = texts[i:i+batch_size]
                    try:
                        batch_emb = model.encode(batch, show_progress_bar=False, convert_to_tensor=False)
                    except TypeError:
                        batch_emb = model.encode(batch, show_progress_bar=False)
                    embeddings.extend(batch_emb)
        except Exception as e:
            log.warning("Re-embedding batches failed, trying single encode: %s", e)
            embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=False)

        embeddings = np.array(embeddings, dtype=np.float32)
        df = pd.DataFrame({
            "doc_id": [c["doc_id"] for c in chunks],
            "pages": [c["pages"] for c in chunks],
            "text": texts,
            "embedding": list(embeddings)
        })
        df.to_parquet(EMBEDDINGS_PARQUET, index=False)
        log.info("Saved re-embedded KB to %s", EMBEDDINGS_PARQUET)
        return df