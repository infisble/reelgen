"""Deterministic handling of the spoken line(s).

Core rule of the pipeline: the text that is spoken is NEVER produced by an LLM.
It is extracted from the user input by code and passed to TTS as-is; the LLM only
references lines by index. Then STT is used to prove the audio matches word-for-word.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Leftmost-first scan over the common quote styles: "..."  «...»  “...”  „...“
_QUOTES = re.compile(r'"([^"]+)"|«([^»]+)»|“([^”]+)”|„([^“”]+)[“”]')
_APOSTROPHES = "'’ʼ`´‘"
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def extract_lines(idea: str) -> list[str]:
    """Return quoted fragments of the input, in order, stripped. Empty list if none."""
    lines = []
    for m in _QUOTES.finditer(idea):
        text = next(g for g in m.groups() if g is not None).strip()
        if text:
            lines.append(text)
    return lines


def detect_lang(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    cyr = sum(1 for c in letters if _CYRILLIC.match(c))
    return "uk" if cyr / len(letters) > 0.3 else "en"


def normalize_words(text: str) -> list[str]:
    """Case/punctuation-insensitive word list. Apostrophes are dropped (STT renders them inconsistently)."""
    t = unicodedata.normalize("NFKC", text).lower()
    for a in _APOSTROPHES:
        t = t.replace(a, "")
    t = re.sub(r"[^\w\s]", " ", t)
    return t.split()


def _word_edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


@dataclass
class VerbatimCheck:
    ok: bool
    expected: str
    heard: str
    wer: float

    def to_dict(self) -> dict:
        return {"ok": self.ok, "expected": self.expected, "heard": self.heard, "wer": round(self.wer, 3)}


def compare(expected: str, heard: str) -> VerbatimCheck:
    e, h = normalize_words(expected), normalize_words(heard)
    dist = _word_edit_distance(e, h)
    return VerbatimCheck(ok=dist == 0, expected=expected, heard=heard, wer=dist / max(len(e), 1))


def estimate_speech_s(text: str) -> float:
    """Rough speech duration (~2.5 words/s) used to reject inputs that cannot fit into 30 s."""
    return len(normalize_words(text)) / 2.5
