import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config.settings import HF_MODEL_NAME, HF_MAX_NEW_TOKENS, HF_TEMPERATURE
from utils.logging_config import setup_logging
from utils.file_utils import extract_json_like, clean_json_response

log = setup_logging()

class LLMClient:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.setup()

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def call_model_json(self, prompt: str, system_instructions: str = None):
        if system_instructions is None:
            system_instructions = (
                "You are a precise scientific QA assistant. "
                "Use ONLY the provided context. Return a single VALID JSON object and nothing else. "
                "Schema: {answer, answer_value, answer_unit, ref_id, ref_url, supporting_materials, explanation}. "
                "If uncertain, answer with 'Unable to answer with confidence based on the provided documents.'"
            )

        full_prompt = f"{system_instructions}\n\nContext and question:\n{prompt}"

        try:
            response = self.generator(
                full_prompt,
                max_new_tokens=HF_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=HF_TEMPERATURE,
                pad_token_id=self.tokenizer.eos_token_id
            )
            raw_text = response[0]["generated_text"]
            parsed = clean_json_response(raw_text)
            if parsed:
                return parsed, raw_text
            else:
                return None, raw_text
        except Exception as e:
            log.warning("HF model call failed: %s", e)
            return None, None