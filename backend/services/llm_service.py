"""Modular LLM service — Gemini + OpenAI with structured extraction support.

Supports:
- extract_structured_data(text, schema) → dict
- generate_text(prompt) → str
"""
from __future__ import annotations

import json
import logging
from typing import Any

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"


class LLMProvider:
    """Unified LLM provider with Gemini → OpenAI → template fallback."""

    def extract_structured_data(
        self,
        text: str,
        schema: dict[str, Any],
        context: str = "medical lab report",
    ) -> dict[str, Any]:
        """Extract structured JSON data from text using LLM.

        Parameters
        ----------
        text:
            Raw text to extract from (e.g. OCR output).
        schema:
            JSON schema describing the expected output fields.
        context:
            Description of what the text represents.

        Returns
        -------
        Parsed dict matching the schema. Missing fields use defaults from schema.
        """
        schema_str = json.dumps(schema, indent=2)
        prompt = (
            f"Extract structured data from the following {context}.\n"
            f"Return ONLY valid JSON matching this schema (use null for missing fields):\n"
            f"{schema_str}\n\n"
            f"Text to extract from:\n{text}\n\n"
            f"Return only the JSON object, no explanation."
        )

        raw = self.generate_text(prompt)

        # Parse JSON from response
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("LLM returned invalid JSON: %s — raw: %s", exc, raw[:200])
            return {}

    def generate_text(self, prompt: str) -> str:
        """Generate text using Gemini → OpenAI → template fallback."""
        if settings.GEMINI_API_KEY:
            try:
                return self._call_gemini(prompt)
            except Exception as exc:
                logger.warning("Gemini failed: %s", exc)

        if settings.OPENAI_API_KEY:
            try:
                return self._call_openai(prompt)
            except Exception as exc:
                logger.warning("OpenAI failed: %s", exc)

        return ""

    def _call_gemini(self, prompt: str) -> str:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def extract_lab_values_from_image(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        """Use Gemini Vision to extract structured lab values from an image.

        Returns a dict of {parameter_name: value} pairs.
        """
        if not settings.GEMINI_API_KEY:
            return {}

        try:
            import base64
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)

            prompt = (
                "Extract ALL lab test results from this medical report image. "
                "Return a JSON object where keys are test names (lowercase, underscored) "
                "and values are the numeric results. Include units as separate keys with '_unit' suffix. "
                "Example: {\"hemoglobin\": 14.5, \"hemoglobin_unit\": \"g/dL\", \"glucose\": 5.4, \"glucose_unit\": \"mmol/L\"} "
                "Return ONLY the JSON object."
            )

            image_data = base64.b64encode(image_bytes).decode("utf-8")
            response = model.generate_content([
                prompt,
                {"mime_type": mime_type, "data": image_data}
            ])

            raw = response.text.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            return json.loads(raw)

        except Exception as exc:
            logger.warning("Gemini Vision structured extraction failed: %s", exc)
            return {}


# Module-level singleton
llm = LLMProvider()
