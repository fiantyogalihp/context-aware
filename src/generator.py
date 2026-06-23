"""Xiaomi Mimo 2.5 generator wrapper.

The client assumes an OpenAI-compatible chat completions endpoint and keeps
generation deterministic with temperature=0.0.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for candidate in (".env", ".envrc"):
    load_dotenv(os.path.join(ROOT, candidate), override=False)

DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-pro"
PAYG_BASE_URL = "https://api.xiaomimimo.com/v1"
TOKEN_PLAN_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"


SYSTEM_PROMPT = (
    "Anda adalah asisten edukasi kesehatan publik Indonesia resmi. "
    "Jawab HANYA berdasarkan konteks resmi yang disediakan. "
    "JANGAN memberikan diagnosis mandiri, resep obat personal, dosis klinis individual, "
    "atau rekomendasi terapi individual.\n\n"
    "Anda WAJIB memberikan output dalam format JSON murni dengan dua kunci berikut:\n"
    '1. "exact_quote": Salin kutipan kata-per-kata dari konteks yang menjawab pertanyaan. '
    'Jika informasinya tidak ada, isi dengan "Tidak ada".\n'
    '2. "final_answer": Jawaban ringkas Anda (1 paragraf atau daftar pendek) yang disusun '
    'HANYA berdasarkan "exact_quote". Gunakan istilah dan urutan langkah dari konteks, '
    'jangan menambahkan detail baru. '
    'Jika "exact_quote" berisi "Tidak ada", maka "final_answer" WAJIB berisi '
    '"Maaf, informasi tersebut tidak tersedia dalam pedoman resmi. '
    'Silakan hubungi tenaga kesehatan atau kanal resmi untuk informasi lebih lanjut."\n\n'
    "Contoh format output:\n"
    '{"exact_quote": "Teks asli dari konteks...", "final_answer": "Jawaban ringkas..."}'
)


@dataclass
class XiaomiMimoClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "XiaomiMimoClient":
        api_key = os.environ.get("MIMO_API_KEY")
        if not api_key:
            raise RuntimeError("MIMO_API_KEY is required for engine=mimo")
        base_url = os.environ.get("MIMO_BASE_URL") or cls.infer_base_url(api_key)
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=os.environ.get("MIMO_MODEL", DEFAULT_MODEL),
        )

    @staticmethod
    def infer_base_url(api_key: str) -> str:
        if api_key.startswith("tp-"):
            return TOKEN_PLAN_BASE_URL
        return PAYG_BASE_URL

    def inject_breadcrumb_to_context(self, retrieved_chunks: Sequence[Dict]) -> str:
        """
        Menyuntikkan jalur hierarki (breadcrumb) ke setiap teks atau tabel 
        sebelum digabungkan dan dikirim ke LLM untuk mengurangi hallucination.
        """
        formatted_contexts = []
        
        for chunk in retrieved_chunks:
            # Build breadcrumb from hierarchy_path or fallback to manual construction
            hierarchy_path = chunk.get("hierarchy_path", [])
            if hierarchy_path:
                breadcrumb = " > ".join(str(p) for p in hierarchy_path if p)
            else:
                # Fallback: construct from title, section, subsection
                parts = [
                    chunk.get("title"),
                    chunk.get("section"),
                    chunk.get("subsection") or chunk.get("subtitle"),
                ]
                breadcrumb = " > ".join(str(p) for p in parts if p)
            
            breadcrumb_header = f"[{breadcrumb}]" if breadcrumb else "[Dokumen]"
            
            # Add explicit table marker for table chunks
            if chunk.get("chunk_type") == "table" or chunk.get("metadata", {}).get("chunk_type") == "table":
                context_text = f"{breadcrumb_header}\nTabel Referensi:\n{chunk.get('content') or chunk.get('text', '')}"
            else:
                context_text = f"{breadcrumb_header}\n{chunk.get('content') or chunk.get('text', '')}"
                
            formatted_contexts.append(context_text)
            
        # Gabungkan semua chunk yang sudah diformat dengan dua baris baru
        return "\n\n".join(formatted_contexts)

    def build_messages(
        self, question: str, context: Sequence[Dict]
    ) -> List[Dict[str, str]]:
        # Use breadcrumb injection instead of raw JSON
        formatted_context = self.inject_breadcrumb_to_context(context)
        return [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\nKonteks Resmi:\n{formatted_context}",
            },
            {"role": "user", "content": question},
        ]

    def generate_answer_with_quote(self, question: str, context: Sequence[Dict]) -> Dict[str, str]:
        """
        Generate answer with exact quote extraction for debugging and attribution tracking.
        Returns dict with 'exact_quote' and 'final_answer' keys.
        """
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self.build_messages(question, context),
            "temperature": 0.0,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 401:
                raise RuntimeError(
                    "MiMo returned 401 Unauthorized. Check that MIMO_API_KEY is a valid "
                    "key from the MiMo console, and that the base URL matches the key type "
                    f"({self.base_url}). Pay-as-you-go keys usually start with sk-, while "
                    "Token Plan keys usually start with tp-."
                ) from exc
            raise
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]
        
        try:
            parsed = json.loads(raw_content)
            return {
                "exact_quote": parsed.get("exact_quote", ""),
                "final_answer": parsed.get("final_answer", raw_content),
            }
        except json.JSONDecodeError:
            return {
                "exact_quote": "",
                "final_answer": raw_content,
            }

    def generate_answer(self, question: str, context: Sequence[Dict]) -> str:
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self.build_messages(question, context),
            "temperature": 0.0,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},  # Force JSON mode
        }
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 401:
                raise RuntimeError(
                    "MiMo returned 401 Unauthorized. Check that MIMO_API_KEY is a valid "
                    "key from the MiMo console, and that the base URL matches the key type "
                    f"({self.base_url}). Pay-as-you-go keys usually start with sk-, while "
                    "Token Plan keys usually start with tp-."
                ) from exc
            raise
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]
        
        # Parse JSON response and extract final_answer
        try:
            parsed = json.loads(raw_content)
            # Return only final_answer for backward compatibility with evaluator
            return parsed.get("final_answer", raw_content)
        except json.JSONDecodeError:
            # Fallback: return raw content if JSON parsing fails
            return raw_content
