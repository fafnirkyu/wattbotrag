import pandas as pd
import torch
import numpy as np
from transformers.utils import logging as hf_logging
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

hf_logging.set_verbosity_error()

from config.settings import *
from utils.logging_config import setup_logging, silence_pdf_warnings
from utils.file_utils import tokenize_len
from data_processing.pdf_processor import download_pdfs
from data_processing.chunking import extract_chunks
from embedding.embedders import Embedder
from embedding.faiss_manager import build_faiss
from retrieval.hybrid_search import HybridSearcher
from generation.llm_client import LLMClient
from generation.answer_generator import AnswerGenerator

def main():
    silence_pdf_warnings()
    log = setup_logging()
    log.info("Starting RAG pipeline")

    # Load data
    metadata = pd.read_csv(METADATA_FILE, encoding="latin1")
    test_questions = pd.read_csv(TEST_Q_FILE, encoding="latin1")
    test_questions.columns = [c.lower().strip() for c in test_questions.columns]
    if "id" not in test_questions.columns or "question" not in test_questions.columns:
        test_questions = test_questions.rename(columns={test_questions.columns[0]: "id", test_questions.columns[1]: "question"})

    # Initialize tokenizer for chunking
    try:
        retriever_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_BASE)
    except Exception:
        retriever_tokenizer = None

    # Step 0: Download PDFs
    log.info("Checking/downloading PDFs...")
    download_pdfs(metadata)

    # Step 1: Extract chunks
    log.info("Extracting chunks...")
    chunks = extract_chunks(metadata, lambda x: tokenize_len(x, retriever_tokenizer), CHUNK_TOKENS, parallel_workers=4)
    if not chunks:
        log.error("No chunks extracted; exiting.")
        return

    # Step 2: Embed chunks
    embedder = Embedder()
    df_chunks = None
    if Path(EMBEDDINGS_PARQUET).exists():
        try:
            log.info("Loading cached embeddings from %s", EMBEDDINGS_PARQUET)
            df_chunks = pd.read_parquet(EMBEDDINGS_PARQUET)
        except Exception:
            df_chunks = None

    if df_chunks is None:
        log.info("Embedding chunks (this may take a while)...")
        df_chunks = embedder.embed_chunks(chunks, model_name=RETRIEVER_BASE, device="cuda" if torch.cuda.is_available() else "cpu")

    # Step 3: Build FAISS index
    log.info("Building FAISS index...")
    index = build_faiss(df_chunks)

    # Step 4: Setup models
    log.info("Setting up models...")
    query_encoder = SentenceTransformer(RETRIEVER_BASE, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Reranker
    try:
        reranker = CrossEncoder(RERANKER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        log.warning("Failed to load reranker %s: %s. Continuing without reranker.", RERANKER_MODEL, e)
        reranker = None

    # LLM and answer generator
    llm_client = LLMClient()
    answer_generator = AnswerGenerator(llm_client, embedder, metadata)
    
    # Hybrid search
    searcher = HybridSearcher(embedder)

    # Answer loop
    log.info("Generating answers with hybrid RAG...")
    all_answers = []

    batch_size = 8
    num_batches = (len(test_questions) + batch_size - 1) // batch_size

    with tqdm(total=num_batches, desc="Answering (batched)", unit="batch") as pbar:
        for i in range(0, len(test_questions), batch_size):
            batch = test_questions.iloc[i:i+batch_size]

            for _, row in batch.iterrows():
                qid, question = row["id"], row["question"]

                try:
                    retrieved = searcher.search(
                        question,
                        query_encoder,
                        df_chunks,
                        index,
                        k=TOP_K,
                        rerank_model=reranker,
                        rerank_top=TOP_K_RERANK
                    )
                    ans_json = answer_generator.generate_answer(question, retrieved, query_encoder)
                except Exception as e:
                    log.error("Error on question %s: %s", qid, e)
                    ref_ids, ref_urls = [], []
                    ans_json = {
                        "answer": "Unable to answer due to pipeline error",
                        "answer_value": "Unknown",
                        "answer_unit": "Unknown",
                        "ref_id": ref_ids,
                        "ref_url": ref_urls,
                        "supporting_materials": "N/A",
                        "explanation": f"Pipeline error: {e}"
                    }

                ans_json["id"] = qid
                ans_json["question"] = question
                all_answers.append(ans_json)

            pbar.update(1)

    # Create submission
    submission_df = pd.DataFrame(all_answers)
    submission_df = submission_df.where(pd.notnull(submission_df), "is_blank")
    for col in ["answer", "answer_value", "answer_unit", "supporting_materials", "explanation"]:
        submission_df[col] = submission_df[col].replace(["", "Unknown", None, np.nan, "N/A"], "is_blank")
    
    def normalize_list_field(x):
        if x in ["", "Unknown", None, np.nan, "N/A", "is_blank"]:
            return "is_blank"
        if isinstance(x, list):
            return ",".join(str(i) for i in x if i)
        return str(x)
    
    submission_df["ref_id"] = submission_df["ref_id"].apply(normalize_list_field)
    submission_df["ref_url"] = submission_df["ref_url"].apply(normalize_list_field)
    submission_df = submission_df.fillna("is_blank").astype(str)
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    log.info("Submission saved: %s", SUBMISSION_FILE)

    blank_count = sum(1 for a in all_answers if str(a.get("answer")).strip().lower() in ["", "unknown", "n/a", "is_blank"])
    log.info("Blank answers: %d/%d (%.2f%%)", blank_count, len(all_answers), blank_count / len(all_answers) * 100)

if __name__ == "__main__":
    main()