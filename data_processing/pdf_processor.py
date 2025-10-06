import requests
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from config.settings import PDF_DIR
from utils.logging_config import setup_logging

log = setup_logging()

def download_pdfs(metadata: pd.DataFrame):
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Downloading PDFs"):
        doc_id, url = row["id"], row.get("url", "")
        pdf_path = PDF_DIR / f"{doc_id}.pdf"
        if pdf_path.exists():
            continue
        if not url or str(url).strip() == "":
            log.warning("No URL for %s", doc_id)
            continue
        try:
            r = requests.get(url, timeout=30, stream=True)
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "").lower()
            if not (str(url).endswith(".pdf") or "pdf" in content_type):
                log.warning("Skipping non-PDF for %s: %s (%s)", doc_id, url, content_type)
                continue
            with open(pdf_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            log.info("Saved PDF: %s", pdf_path)
        except Exception as e:
            log.warning("Failed to download %s: %s", doc_id, e)