import re
import pdfplumber
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from config.settings import PDF_DIR
from utils.logging_config import setup_logging

log = setup_logging()

def chunk_text(pages_text: List[Tuple[int, str]], doc_id: str, tokenize_len_func, CHUNK_TOKENS: int) -> List[Dict[str, Any]]:
    chunks = []
    current = ""
    current_pages = []
    for pnum, text in pages_text:
        sentences = re.split(r'(?<=[\.\?\!])\s+', (text or "").strip())
        for s in sentences:
            if not s.strip():
                continue
            joined = (current + " " + s).strip()
            if tokenize_len_func(joined) <= CHUNK_TOKENS or current == "":
                current = joined
                current_pages.append(pnum)
            else:
                chunks.append({"doc_id": str(doc_id), "pages": list(set(current_pages)), "text": current.strip()})
                current = s.strip()
                current_pages = [pnum]
    if current.strip():
        chunks.append({"doc_id": str(doc_id), "pages": list(set(current_pages)), "text": current.strip()})
    return chunks

def _process_pdf_row(row, tokenize_len_func, CHUNK_TOKENS: int):
    doc_id = row['id']
    pdf_path = PDF_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        log.warning("PDF missing for %s — skipping", doc_id)
        return []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [(i+1, page.extract_text() or "") for i, page in enumerate(pdf.pages)]
            return chunk_text(pages, doc_id, tokenize_len_func, CHUNK_TOKENS)
    except Exception as e:
        log.warning("Error reading %s: %s", pdf_path, e)
        return []

def extract_chunks(metadata: pd.DataFrame, tokenize_len_func, CHUNK_TOKENS: int, parallel_workers: int = 4) -> List[Dict[str, Any]]:
    all_chunks = []
    rows = [row for _, row in metadata.iterrows()]
    with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
        for chunk_list in tqdm(ex.map(lambda row: _process_pdf_row(row, tokenize_len_func, CHUNK_TOKENS), rows), 
                              total=len(rows), desc="Extracting chunks"):
            if chunk_list:
                all_chunks.extend(chunk_list)
    log.info("Extracted %d chunks", len(all_chunks))
    return all_chunks