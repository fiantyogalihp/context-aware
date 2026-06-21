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
    "Gunakan istilah dan urutan langkah dari konteks, jangan menambahkan detail baru, "
    "dan buat jawaban ringkas dalam 1 paragraf atau daftar pendek. "
    "Jangan memberikan diagnosis mandiri, resep obat personal, dosis klinis individual, "
    "atau rekomendasi terapi individual. Jika konteks tidak cukup, nyatakan bahwa informasi "
    "tidak tersedia dalam konteks resmi dan arahkan pengguna ke tenaga kesehatan atau kanal resmi. "
    "Untuk pertanyaan prosedur aplikasi, jawab hanya langkah yang eksplisit ada di konteks."
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

    def build_messages(
        self, question: str, context: Sequence[Dict]
    ) -> List[Dict[str, str]]:
        context_json = json.dumps(list(context), ensure_ascii=False, indent=2)
        return [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\nKonteks Resmi:\n{context_json}",
            },
            {"role": "user", "content": question},
        ]

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
        return data["choices"][0]["message"]["content"]
