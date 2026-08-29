"""Embedding layer: pluggable, zero-dependency by default.

The default :class:`HashingEmbedder` builds a fixed-dimension vector from
character n-grams with TF-IDF-like weighting, so the library works out of the
box without any model downloads or API keys. It is deliberately simple; for
production quality you can swap in a `SentenceTransformerEmbedder` (via the
`sentence-transformers` extra) or any custom embedder exposing ``embed(text)``.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Duck-typed embedder interface."""

    dim: int

    def embed(self, text: str) -> np.ndarray: ...


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HashingEmbedder:
    """Hashing embedder with IDF weighting: character n-grams + word tokens.

    * No external models, downloads or API keys.
    * Works for both Chinese (character n-grams) and English (word tokens).
    * Word tokens carry a weight multiplier — they carry more semantics than
      raw character windows ("indentation" vs "indent" finally overlap).
    * Uses sub-linear TF and corpus-level IDF for better discriminability.
    """

    def __init__(
        self,
        dim: int = 512,
        ngram_range: tuple[int, int] = (1, 3),
        use_words: bool = True,
        word_repeats: int = 3,
        smooth_idf: float = 1.0,
    ):
        self.dim = dim
        self.ngram_range = ngram_range
        self.use_words = use_words
        self.word_repeats = word_repeats
        self.smooth_idf = smooth_idf
        self._doc_freq = Counter()
        self._n_docs = 0

    def _features(self, text: str) -> list[str]:
        text = _normalize(text)
        if not text:
            return []
        feats: list[str] = []
        chars = list(text)
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            feats.extend("".join(chars[i : i + n]) for i in range(len(chars) - n + 1))
        if self.use_words:
            for w in re.findall(r"[a-z0-9]+", text):
                if len(w) >= 2:
                    # "w:" prefix avoids collisions with character n-grams;
                    # repeated occurrences up-weight the word's contribution.
                    feats.extend([f"w:{w}"] * self.word_repeats)
        return feats

    def _hash_index(self, gram: str) -> int:
        # Deterministic 64-bit hash (FNV-1a) mapped into [0, dim)
        h = 14695981039346656037
        for ch in gram.encode("utf-8"):
            h ^= ch
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h % self.dim

    def fit(self, texts: list[str]) -> "HashingEmbedder":
        """Compute IDF statistics over a corpus."""
        seen: set[str] = set()
        for t in texts:
            seen.clear()
            for g in self._features(t):
                if g not in seen:
                    seen.add(g)
                    self._doc_freq[g] += 1
            self._n_docs += 1
        return self

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts = Counter(self._features(text))
        n = sum(counts.values()) or 1
        for gram, c in counts.items():
            tf = 1.0 + math.log(c)  # sub-linear TF
            idx = self._hash_index(gram)
            df = self._doc_freq.get(gram, 0)
            idf = 1.0
            if self._n_docs:
                idf = math.log((self._n_docs + self.smooth_idf) / (df + self.smooth_idf)) + 1.0
            vec[idx] += (tf / n) * idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def __repr__(self) -> str:
        return f"HashingEmbedder(dim={self.dim}, ngram={self.ngram_range})"


class SentenceTransformerEmbedder:
    """Optional embedding backend backed by sentence-transformers.

    Requires ``pip install memstale[st]``.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install memstale[st]"
            ) from exc
        self._model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)
