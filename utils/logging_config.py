import logging
from pathlib import Path

def setup_logging():
    LOGFILE = Path("logs/wattbot_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOGFILE),
            logging.StreamHandler()
        ],
    )
    return logging.getLogger("wattbot")

# Silence PDF warnings
def silence_pdf_warnings():
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("pdfplumber").setLevel(logging.ERROR)
    logging.getLogger("pdfminer.layout").setLevel(logging.ERROR)
    logging.getLogger("pdfminer.pdfinterp").setLevel(logging.ERROR)
    logging.getLogger("pdfminer.converter").setLevel(logging.ERROR)