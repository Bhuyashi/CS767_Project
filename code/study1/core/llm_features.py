"""LLM-based metric scoring for Study 1 radiology reports.

Supports three backends:
  - gemini        : Google Gemini API (google-generativeai)
  - ollama        : Local Llama via Ollama (http://localhost:11434)
  - openai_compat : Any OpenAI-compatible endpoint (Together AI, Groq, local vLLM)

Each backend implements ``score(report_text)`` → dict[str, float] | None.
Returns None on unrecoverable parse/API failure after retries.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Literal

logger = logging.getLogger(__name__)

LLM_METRIC_COLUMNS: list[str] = [
    "specificity",
    "abbreviation_usage",
    "hedge_rate",
    "urgency_signaling",
    "actionable_recommendation_rate",
]

PROMPT_TEMPLATE = """\
You are an expert radiologist and medical language analyst. You will be given a radiology report and must score it on 5 metrics. Each score must be a number between 0 and 1 (inclusive), rounded to 2 decimal places.

---

## METRIC DEFINITIONS

### 1. Specificity (detailed)
Measures how precisely findings are described in terms of measurements, anatomical locations, and characteristics.
- **0.0** — No measurements, vague locations (e.g., "there is an opacity"), no descriptive characteristics
- **0.5** — Some specificity; approximate locations given for some findings
- **1.0** — All major findings include precise anatomical location (e.g., "right lower lobe, posterior segment"), and descriptive characteristics (e.g., "spiculated margins, heterogeneous density")

### 2. Abbreviation Usage
Measures how frequently abbreviations are used relative to the total number of terms that could have been abbreviated or spelled out.
- **0.0** — No abbreviations used; all terms are fully spelled out
- **0.5** — Moderate use; abbreviations appear occasionally alongside spelled-out terms
- **1.0** — Heavy use of abbreviations throughout the report (e.g., "SOB," "RLL," "LAD," "HTN," "r/o") with minimal spelled-out equivalents

### 3. Hedge Rate / Low Confidence
Measures how frequently uncertain or hedging language is used, and whether the report relies on vague qualifiers instead of confident assertions.
Hedge words and phrases include: *may*, *might*, *could*, *possibly*, *probable*, *likely*, *cannot exclude*, *suspicious for*, *questionable*, *appears to*, *seems*, *cannot be ruled out*, *differential includes*.
- **0.0** — No hedging language; all findings stated with full confidence and directness
- **0.5** — Hedging is present but balanced; used selectively for genuinely uncertain findings
- **1.0** — Pervasive hedging throughout; nearly every finding is qualified with uncertain language even where confidence should be high

### 4. Urgency Signaling
Measures whether the report communicates time-sensitive findings clearly and explicitly.
Urgency indicators include words/phrases like: *urgent*, *emergent*, *immediate*, *STAT*, *critical*, *requires prompt attention*, *notify clinician*, *do not delay*, *life-threatening*, *acute*.
- **0.0** — No urgency language present, even if urgent findings exist
- **0.5** — Mild urgency implied (e.g., "follow-up recommended soon") but not stated explicitly
- **1.0** — Urgent findings are clearly flagged with explicit urgency language and direct communication instructions

### 5. Actionable Recommendation Rate
Measures how often the report's impression or conclusion section includes a concrete, specific next step for the clinician.
Actionable recommendations include: ordering specific follow-up imaging, recommending biopsy, suggesting clinical correlation with a named test, specifying a follow-up timeframe (e.g., "repeat CT in 3 months").
Non-actionable language includes: "clinical correlation suggested," "findings noted," "discussed with clinician."
- **0.0** — No actionable recommendations anywhere in the report
- **0.5** — At least one vague recommendation present, but lacking specificity (e.g., "follow-up imaging may be considered")
- **1.0** — Every significant finding is paired with a clear, specific, and actionable next step

---

## OUTPUT FORMAT

Return ONLY a valid JSON object in the following format. Do not include any explanation or additional text.

{{
  "specificity": <0.0-1.0>,
  "abbreviation_usage": <0.0-1.0>,
  "hedge_rate": <0.0-1.0>,
  "urgency_signaling": <0.0-1.0>,
  "actionable_recommendation_rate": <0.0-1.0>
}}

---

## RADIOLOGY REPORT

{report_text}"""


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _validate_scores(parsed: dict) -> dict[str, float] | None:
    result: dict[str, float] = {}
    for col in LLM_METRIC_COLUMNS:
        val = parsed.get(col)
        if val is None:
            logger.debug("Missing metric %r in LLM response", col)
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= f <= 1.0):
            logger.debug("Metric %r out of range: %s", col, val)
            return None
        result[col] = round(f, 2)
    return result


class LLMScorer(ABC):
    @abstractmethod
    def _call_api(self, prompt: str) -> str: ...

    def score(
        self,
        report_text: str,
        *,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> dict[str, float] | None:
        prompt = PROMPT_TEMPLATE.format(report_text=report_text.strip())
        for attempt in range(retries):
            try:
                raw = self._call_api(prompt)
                parsed = _extract_json(raw)
                if parsed is None:
                    logger.warning("JSON parse failed (attempt %d/%d)", attempt + 1, retries)
                    continue
                validated = _validate_scores(parsed)
                if validated is None:
                    logger.warning("Invalid metric values (attempt %d/%d): %s", attempt + 1, retries, parsed)
                    continue
                return validated
            except Exception as exc:
                logger.warning("API error (attempt %d/%d): %s", attempt + 1, retries, exc)
                if attempt < retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
        return None


class GeminiScorer(LLMScorer):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        logger.info("GeminiScorer ready: model=%s", model)

    def _call_api(self, prompt: str) -> str:
        return self._model.generate_content(prompt).text


class OllamaScorer(LLMScorer):
    """Llama (or any model) via a local Ollama server."""
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")
        logger.info("OllamaScorer ready: model=%s base_url=%s", model, self._base_url)

    def _call_api(self, prompt: str) -> str:
        import urllib.request
        payload = json.dumps({"model": self._model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode())["response"]


class OpenAICompatibleScorer(LLMScorer):
    """Any OpenAI-compatible endpoint: Together AI, Groq, local vLLM, etc."""
    def __init__(self, api_key: str, base_url: str, model: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        logger.info("OpenAICompatibleScorer ready: model=%s base_url=%s", model, base_url)

    def _call_api(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content


class HuggingFaceScorer(LLMScorer):
    """Local HuggingFace model via transformers pipeline (GPU-accelerated).

    Recommended models (no gated access required):
      microsoft/Phi-3-mini-4k-instruct     — 3.8B, fits 8GB GPU, great instruction following
      aaditya/Llama3-OpenBioLLM-8B         — 8B medical, fits A10 (24GB) or 8GB with 4-bit
      BioMistral/BioMistral-7B-DARE        — 7B medical, fits A10 or 8GB with 4-bit

    quantization: "4bit" or "8bit" reduces VRAM usage (requires bitsandbytes):
        pip install bitsandbytes
        "4bit" : ~4-5GB for a 7B model  (RTX 2080 compatible)
        "8bit" : ~7-8GB for a 7B model
        "none" : full float16, ~14GB for 7B (A10 / A100 recommended)
    """

    def __init__(
        self,
        model: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "auto",
        quantization: str = "none",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
        except ImportError:
            raise ImportError("pip install transformers accelerate")

        quantization_config = None
        if quantization in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError("pip install bitsandbytes  # required for 4-bit/8-bit quantization")
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(
            "Loading HuggingFace model: %s  (device_map=%s, quantization=%s)",
            model, device, quantization,
        )

        kwargs: dict = {"device_map": device, "torch_dtype": "auto"}
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config

        self._pipe = hf_pipeline(
            "text-generation",
            model=model,
            **kwargs,
        )
        self._model_name = model
        logger.info("HuggingFaceScorer ready: model=%s quantization=%s", model, quantization)

    def _call_api(self, prompt: str) -> str:
        outputs = self._pipe(
            [{"role": "user", "content": prompt}],
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
        )
        # pipeline returns [{"generated_text": [{"role": ..., "content": ...}]}]
        result = outputs[0]["generated_text"]
        if isinstance(result, list):
            return str(result[-1].get("content", result[-1]))
        return str(result)


BackendType = Literal["gemini", "ollama", "openai_compat", "hf"]


def build_scorer(
    backend: BackendType,
    *,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    quantization: str = "none",
) -> LLMScorer:
    if backend == "gemini":
        return GeminiScorer(api_key=api_key, model=model or "gemini-1.5-flash")
    if backend == "ollama":
        return OllamaScorer(model=model or "llama3", base_url=base_url or "http://localhost:11434")
    if backend == "openai_compat":
        if not base_url or not model:
            raise ValueError("--base-url and --model are required for openai_compat backend")
        return OpenAICompatibleScorer(api_key=api_key, base_url=base_url, model=model)
    if backend == "hf":
        return HuggingFaceScorer(
            model=model or "microsoft/Phi-3-mini-4k-instruct",
            quantization=quantization,
        )
    raise ValueError(f"Unknown backend: {backend!r}. Choose: gemini, ollama, openai_compat, hf")
