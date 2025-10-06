import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logging_config import setup_logging

log = setup_logging()

class HybridSearcher:
    def __init__(self, embedder):
        self.embedder = embedder

    def hybrid_search(self, query: str, model: SentenceTransformer, df: pd.DataFrame, index, k: int, tfidf_top: int):
        try:
            qvec = model.encode([query], convert_to_tensor=False)
        except TypeError:
            qvec = model.encode([query])
        qvec = np.array(qvec, dtype=np.float32).reshape(1, -1)
        D, I = index.search(qvec, k)
        vec_idx = I[0].tolist()
        vec_candidates = df.iloc[vec_idx].copy().reset_index(drop=True)

        tfidf_candidates = pd.DataFrame()
        if self.embedder.tfidf_vectorizer is not None and self.embedder.tfidf_matrix is not None:
            q_tfidf = self.embedder.tfidf_vectorizer.transform([query])
            sims = cosine_similarity(q_tfidf, self.embedder.tfidf_matrix).flatten()
            top_idx = sims.argsort()[::-1][:tfidf_top]
            tfidf_candidates = df.iloc[top_idx].copy().reset_index(drop=True)
            tfidf_candidates["tfidf_sim"] = sims[top_idx]

        seen = set()
        merged_rows = []
        for _, r in vec_candidates.iterrows():
            key = (str(r.doc_id), r.text[:120])
            if key not in seen:
                merged_rows.append(r)
                seen.add(key)
        for _, r in tfidf_candidates.iterrows():
            key = (str(r.doc_id), r.text[:120])
            if key not in seen:
                merged_rows.append(r)
                seen.add(key)

        if len(merged_rows) == 0:
            return pd.DataFrame(columns=df.columns)
        merged_df = pd.DataFrame(merged_rows).reset_index(drop=True)
        return merged_df

    def search(self, query: str, model: SentenceTransformer, df: pd.DataFrame, index, k: int,
               rerank_model=None, rerank_top: int = 5):
        candidates = self.hybrid_search(query, model, df, index, k=k, tfidf_top=k)
        if rerank_model is not None and len(candidates) > 0:
            pairs = [[query, t] for t in candidates["text"].tolist()]
            try:
                scores = rerank_model.predict(pairs)
                candidates["score"] = scores
                candidates = candidates.sort_values("score", ascending=False).head(rerank_top).reset_index(drop=True)
            except Exception as e:
                log.warning("Reranker failed: %s", e)
        return candidates