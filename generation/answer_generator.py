import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import TOP_K_RERANK
from utils.logging_config import setup_logging
from utils.file_utils import extract_json_like

log = setup_logging()

class AnswerGenerator:
    def __init__(self, llm_client, embedder, metadata):
        self.llm_client = llm_client
        self.embedder = embedder
        self.metadata = metadata

    def extract_answer_from_raw(self, raw_text: str):
        parsed = extract_json_like(raw_text)
        if parsed and parsed.get("answer"):
            return parsed
        txt = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        sents = re.split(r'(?<=[\.\?\!])\s+', txt)
        candidates = []
        for s in sents:
            if re.search(r"\d", s) or re.search(r"\bpercent\b|\b%\b|\bkg\b|\btons?\b|\btonne\b|\bmetric\b", s, flags=re.I):
                candidates.append(s.strip())
        if candidates:
            return {"answer": " ".join(candidates[:2]), "answer_value": "Unknown", "answer_unit": "Unknown"}
        if sents:
            return {"answer": sents[0].strip(), "answer_value": "Unknown", "answer_unit": "Unknown"}
        return None

    def best_chunk_match(self, text: str, candidates: pd.DataFrame):
        if self.embedder.tfidf_vectorizer is None or self.embedder.tfidf_matrix is None or candidates.empty:
            return None
        qv = self.embedder.tfidf_vectorizer.transform([text])
        best_sim = -1.0
        best_idx = None
        for i, ct in enumerate(candidates["text"].tolist()):
            try:
                ctv = self.embedder.tfidf_vectorizer.transform([ct])
                s = cosine_similarity(qv, ctv).flatten()[0]
                if s > best_sim:
                    best_sim = s
                    best_idx = i
            except Exception:
                continue
        return best_idx

    def generate_answer(self, question: str, retrieved_chunks: pd.DataFrame, query_encoder: SentenceTransformer):
        combined_text = "\n\n".join([f"[{row.doc_id}] {row.text}" for _, row in retrieved_chunks.iterrows()])
        prompt = f"Question: {question}\n\nContext:\n{combined_text}"

        parsed, raw = self.llm_client.call_model_json(prompt)

        if parsed:
            keys = ["answer","answer_value","answer_unit","ref_id","ref_url","supporting_materials","explanation"]
            result = {k: parsed.get(k, "N/A") for k in keys}
            ref_ids = result.get("ref_id") or []
            if isinstance(ref_ids, str):
                ref_ids = [r.strip() for r in ref_ids.split(",") if r.strip()]
            if not isinstance(ref_ids, (list, tuple)):
                ref_ids = [str(ref_ids)]
            ref_ids = [r for r in ref_ids if r not in ["N/A", "Unknown", "", None]]
            if not ref_ids and len(retrieved_chunks) > 0:
                ref_ids = list(map(str, retrieved_chunks["doc_id"].tolist()[:TOP_K_RERANK]))
            result["ref_id"] = ref_ids

            urls = []
            for rid in ref_ids:
                match = self.metadata.loc[self.metadata["id"] == rid, "url"].tolist()
                if match:
                    urls.extend(match)
            if not urls and len(retrieved_chunks) > 0:
                u = self.metadata.loc[self.metadata["id"] == retrieved_chunks.iloc[0].doc_id, "url"].tolist()
                urls = u if u else ["N/A"]
            result["ref_url"] = urls or ["N/A"]

            for k in ["answer","answer_value","answer_unit","supporting_materials","explanation"]:
                if not result.get(k):
                    result[k] = "N/A"
            return result

        extracted = self.extract_answer_from_raw(raw or "")
        if extracted:
            best_idx = self.best_chunk_match(extracted.get("answer", question), retrieved_chunks)
            supporting, ref_id_list, ref_url_list = None, [], []
            if best_idx is not None and best_idx < len(retrieved_chunks):
                best_row = retrieved_chunks.iloc[best_idx]
                supporting = best_row.text
                ref_id_list = [str(best_row.doc_id)]
                ref_url_list = self.metadata.loc[self.metadata['id'] == best_row.doc_id, 'url'].tolist() or ["N/A"]
            else:
                supporting = combined_text[:1000] if combined_text else "N/A"
                ref_id_list = list(map(str, retrieved_chunks["doc_id"].tolist()[:TOP_K_RERANK])) if len(retrieved_chunks)>0 else []
                ref_url_list = [self.metadata.loc[self.metadata['id'] == r, 'url'].iloc[0] if r in self.metadata['id'].values else "N/A" for r in ref_id_list] or ["N/A"]

            out = {
                "answer": extracted.get("answer", "N/A"),
                "answer_value": extracted.get("answer_value", "Unknown"),
                "answer_unit": extracted.get("answer_unit", "Unknown"),
                "ref_id": ref_id_list,
                "ref_url": ref_url_list,
                "supporting_materials": supporting or "N/A",
                "explanation": "Extracted from model output and mapped to best retrieved document(s)."
            }
            return out

        if len(retrieved_chunks) > 0:
            ref_ids = list(map(str, retrieved_chunks["doc_id"].tolist()[:TOP_K_RERANK]))
            ref_urls = [self.metadata.loc[self.metadata['id'] == r, 'url'].iloc[0] if r in self.metadata['id'].values else "N/A" for r in ref_ids]
            fallback = {
                "answer": "Unable to confidently answer from the provided documents. See retrieved documents for details.",
                "answer_value": "Unknown",
                "answer_unit": "Unknown",
                "ref_id": ref_ids,
                "ref_url": ref_urls or ["N/A"],
                "supporting_materials": combined_text[:1000] if combined_text else "N/A",
                "explanation": "Model did not produce a parsable answer; returning retrieved evidence for manual inspection."
            }
            return fallback

        return {
            "answer": "No supporting documents available to answer the question.",
            "answer_value": "Unknown",
            "answer_unit": "Unknown",
            "ref_id": [],
            "ref_url": ["N/A"],
            "supporting_materials": "N/A",
            "explanation": "No documents were retrieved by RAG."
        }