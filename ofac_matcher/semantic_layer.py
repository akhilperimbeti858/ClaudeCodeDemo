"""
Semantic vector similarity layer.

Primary backend  — sentence-transformers (dense transformer embeddings).
Fallback backend — TF-IDF with character n-grams (works fully offline,
                   no model download required).

Both backends produce L2-normalised vectors so the rest of the pipeline
(cosine similarity, L1/L2 distance) is identical regardless of backend.

Two distances are exposed:
  L1 (Manhattan) – sum of absolute element-wise differences.
                   More robust to outlier dimensions.
  L2 (Euclidean) – standard geometric distance.  For unit vectors:
                       cosine_sim = 1 - (L2² / 2)

The combined semantic score returned to the pipeline is cosine similarity
[0, 1]; the raw L1/L2 distance is preserved in LayerScore for transparency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import DistanceMetric, SDNEntity

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Embedder ABC + two concrete backends
# ---------------------------------------------------------------------------

class _Embedder:
    """Minimal interface both backends must satisfy."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 array of shape (len(texts), D)."""
        raise NotImplementedError

    @property
    def backend_name(self) -> str:
        raise NotImplementedError


class _SentenceTransformerEmbedder(_Embedder):
    def __init__(self, model_name: str, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    @property
    def backend_name(self) -> str:
        return f"sentence-transformers/{self._model_name}"


class _TFIDFEmbedder(_Embedder):
    """
    Character n-gram TF-IDF embedder.

    Trained lazily on the first call to ``encode()`` (or explicitly via
    ``fit()``).  Good at capturing shared substrings, transliterations, and
    partial name overlaps — exactly the kind of variation seen in OFAC aliases.
    """

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
            max_features=8192,
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            # Lazy fit on the first batch — not ideal, but safe for ad-hoc use
            self.fit(texts)
        mat = self._vectorizer.transform(texts).toarray().astype(np.float32)
        # L2-normalise each row
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms

    @property
    def backend_name(self) -> str:
        return "tfidf-char-ngram"


# ---------------------------------------------------------------------------
# SemanticCandidate + SemanticIndex
# ---------------------------------------------------------------------------

@dataclass
class SemanticCandidate:
    sdn_entity: SDNEntity
    cosine_similarity: float   # [0, 1]
    distance: float            # raw L1 or L2 value
    matched_alias: str


class SemanticIndex:
    """
    Holds pre-computed embeddings for every SDN name / alias and supports
    fast nearest-neighbour lookups.

    Parameters
    ----------
    model_name:       Sentence-transformers model name or local path.
                      Ignored when ``use_tfidf_fallback=True``.
    distance_metric:  DistanceMetric.L1 or DistanceMetric.L2.
    batch_size:       Batch size for transformer encoding.
    use_tfidf_fallback:
                      Force TF-IDF backend even if sentence-transformers
                      is available.  Useful for offline / test environments.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        batch_size: int = 256,
        use_tfidf_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.batch_size = batch_size

        self._embedder = self._init_embedder(model_name, batch_size, use_tfidf_fallback)
        logger.info("Semantic backend: %s", self._embedder.backend_name)

        self._embeddings: Optional[np.ndarray] = None
        self._name_index: list[tuple[str, SDNEntity]] = []

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @staticmethod
    def _init_embedder(
        model_name: str,
        batch_size: int,
        force_tfidf: bool,
    ) -> _Embedder:
        if force_tfidf:
            return _TFIDFEmbedder()

        try:
            logger.info("Loading sentence-transformer model: %s", model_name)
            return _SentenceTransformerEmbedder(model_name, batch_size)
        except Exception as exc:
            logger.warning(
                "sentence-transformers unavailable (%s). "
                "Falling back to TF-IDF character n-gram embedder.",
                exc,
            )
            return _TFIDFEmbedder()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, sdn_list: list[SDNEntity]) -> None:
        """Embed every primary name and alias in *sdn_list*."""
        name_index: list[tuple[str, SDNEntity]] = []
        for entity in sdn_list:
            for alias in entity.all_names:
                name_index.append((alias, entity))

        if not name_index:
            logger.warning("build_index called with empty SDN list — index will be empty")
            self._name_index = []
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            return

        names = [n for n, _ in name_index]

        # For TF-IDF, pre-fit on the entire SDN corpus first so IDF weights
        # reflect all entity names, not just the first encode() call.
        if isinstance(self._embedder, _TFIDFEmbedder):
            self._embedder.fit(names)

        logger.info("Embedding %d SDN name strings …", len(names))
        self._embeddings = self._embedder.encode(names)
        self._name_index = name_index
        logger.info("Index built: %d vectors of dim %d", *self._embeddings.shape)

    @property
    def is_built(self) -> bool:
        return self._embeddings is not None and self._embeddings.shape[0] > 0

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _l1_distances(self, query_vec: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(self._embeddings - query_vec), axis=1)

    def _l2_distances(self, query_vec: np.ndarray) -> np.ndarray:
        diff = self._embeddings - query_vec
        return np.sqrt(np.sum(diff * diff, axis=1))

    def _cosine_similarities(self, query_vec: np.ndarray) -> np.ndarray:
        sims = self._embeddings @ query_vec   # (N,) — valid since rows are unit-normalised
        return np.clip(sims, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        threshold: float = 0.5,
        candidate_uids: Optional[set[str]] = None,
    ) -> list[SemanticCandidate]:
        """
        Find the *top_k* SDN entries most semantically similar to *query_text*.

        Parameters
        ----------
        query_text:      Entity string from Comprehend.
        top_k:           Maximum results to return.
        threshold:       Minimum cosine similarity [0, 1] to keep.
        candidate_uids:  If provided, restrict search to this UID set
                         (use after fuzzy pre-filter to narrow the search).
        """
        if not self.is_built:
            raise RuntimeError("Call build_index() before querying.")

        query_vec = self._embedder.encode([query_text])[0]  # (D,)

        cosine_sims = self._cosine_similarities(query_vec)

        if self.distance_metric == DistanceMetric.L1:
            distances = self._l1_distances(query_vec)
        else:
            distances = self._l2_distances(query_vec)

        if candidate_uids is not None:
            mask = np.array(
                [self._name_index[i][1].uid in candidate_uids for i in range(len(self._name_index))],
                dtype=bool,
            )
            cosine_sims = np.where(mask, cosine_sims, -1.0)

        n = len(cosine_sims)
        k = min(top_k * 3, n)
        top_indices = np.argpartition(cosine_sims, -k)[-k:]
        top_indices = top_indices[np.argsort(cosine_sims[top_indices])[::-1]]

        seen_uids: set[str] = set()
        results: list[SemanticCandidate] = []

        for idx in top_indices:
            sim = float(cosine_sims[idx])
            if sim < threshold:
                break
            alias, entity = self._name_index[idx]
            if entity.uid in seen_uids:
                continue
            seen_uids.add(entity.uid)
            results.append(
                SemanticCandidate(
                    sdn_entity=entity,
                    cosine_similarity=sim,
                    distance=float(distances[idx]),
                    matched_alias=alias,
                )
            )
            if len(results) >= top_k:
                break

        return results
