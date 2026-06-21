
"""Tiny lexical retriever for demo and tests; not intended as a production vector DB."""
from __future__ import annotations
from typing import Dict, List
import json, re
from pathlib import Path

STOP = {"yang","dan","di","ke","dari","untuk","pada","ini","itu","dengan","atau","sebagai","dalam","akan","dapat","bisa","harus","adalah","ialah","oleh","agar","jika","sudah","lalu","kemudian","maka","anda","pengguna","peserta","fitur","menu","tap","klik","pilih","mobile","jkn","bpjs","kesehatan"}

def tokenize(text: str):
    toks = re.sub(r"[^a-zA-Z0-9À-ÿ]+", " ", text.lower()).split()
    return [t for t in toks if len(t) > 2 and t not in STOP]

def load_jsonl(path: str | Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def retrieve(query: str, chunks: List[Dict], top_k: int = 4) -> List[Dict]:
    q = set(tokenize(query))
    scored = []
    for c in chunks:
        text = " ".join(str(c.get(k,"")) for k in ("title","section","text"))
        toks = set(tokenize(text))
        if not toks:
            score = 0.0
        else:
            score = len(q & toks) / max(1, len(q))
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored[:top_k] if score > 0]
