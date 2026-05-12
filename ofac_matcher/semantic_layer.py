"""
FAISS-backed semantic similarity layer with per-silo indexes.

Architecture
------------
Three FAISS IndexFlatIP indexes are maintained — one per SDNSilo:
  OFAC_ORG  →  organisations, vessels, aircraft
  OFAC_POI  →  individuals / persons of interest
  FTO       →  foreign terrorist organisations (SDGT-programme entities)

Queries are routed to only the silos selected by the Comprehend entity type,
so PERSON queries never touch the ORG/FTO index and vice-versa.

FAISS index type
    IndexFlatIP (inner product) with L2-normalised vectors = cosine similarity.
    This is exact (brute-force) — appropriate for SDN list sizes (~20 k entries).
    For > 500 k entries consider switching to IndexIVFFlat with nprobe tuning.

Distance reporting
    The pipeline reports either L1 (Manhattan) or L2 (Euclidean) distance
    on the top-K FAISS results for transparency, while nearest-neighbour
    traversal is always done via cosine similarity.

Semantic tiers
    STRONG  cosine >= 0.95
    GOOD    cosine >= 0.80
    PARTIAL cosine >= pipeline threshold

Embedding backends
    Primary  : sentence-transformers (any HuggingFace model)
    Fallback : TF-IDF char n-gram (offline, no model download)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

from .models import DistanceMetric, SDNEntity, SDNSilo, SemanticTier

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Cosine similarity thresholds for semantic tier assignment
_TIER_STRONG = 0.95
_TIER_GOOD = 0.80


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class _Embedder:
    def encode(self, texts: list[str]) -> np.ndarray:
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
    """Character n-gram TF-IDF with dense L2-normalised output."""

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
            self.fit(texts)
        mat = self._vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms

    @property
    def backend_name(self) -> str:
        return "tfidf-char-ngram"


# ---------------------------------------------------------------------------
# SemanticCandidate
# ---------------------------------------------------------------------------

@dataclass
class SemanticCandidate:
    sdn_entity: SDNEntity
    cosine_similarity: float    # [0, 1]
    distance: float             # raw L1 or L2 value
    matched_alias: str
    semantic_tier: SemanticTier
    silo: SDNSilo


def _assign_semantic_tier(cosine: float, threshold: float) -> SemanticTier:
    if cosine >= _TIER_STRONG:
        return SemanticTier.STRONG
    if cosine >= _TIER_GOOD:
        return SemanticTier.GOOD
    return SemanticTier.PARTIAL


# ---------------------------------------------------------------------------
# SemanticIndex — three siloed FAISS indexes
# ---------------------------------------------------------------------------

class SemanticIndex:
    """
    FAISS-backed semantic index partitioned into three SDN silos.

    Parameters
    ----------
    model_name:          sentence-transformers model name or local path.
    distance_metric:     DistanceMetric.L1 or DistanceMetric.L2 (for LayerScore).
    batch_size:          Encoder batch size.
    use_tfidf_fallback:  Force TF-IDF backend (no network required).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        distance_metric: DistanceMetric = DistanceMetric.L2,
        batch_size: int = 256,
        use_tfidf_fallback: bool = False,
    ) -> None:
        self.distance_metric = distance_metric
        self._embedder = self._init_embedder(model_name, batch_size, use_tfidf_fallback)
        logger.info("Semantic backend: %s", self._embedder.backend_name)

        # Per-silo FAISS indexes and parallel name→entity mapping
        self._faiss_indexes: dict[SDNSilo, faiss.Index] = {}
        self._name_index: dict[SDNSilo, list[tuple[str, SDNEntity]]] = {}
        # Raw embeddings kept for L1/L2 distance computation post-search
        self._embeddings: dict[SDNSilo, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @staticmethod
    def _init_embedder(model_name: str, batch_size: int, force_tfidf: bool) -> _Embedder:
        if force_tfidf:
            return _TFIDFEmbedder()
        try:
            logger.info("Loading sentence-transformer model: %s", model_name)
            return _SentenceTransformerEmbedder(model_name, batch_size)
        except Exception as exc:
            logger.warning(
                "sentence-transformers unavailable (%s). Falling back to TF-IDF.", exc
            )
            return _TFIDFEmbedder()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, sdn_list: list[SDNEntity]) -> None:
        """
        Embed all SDN names / aliases and build one FAISS IndexFlatIP per silo.
        Entities must already have their ``silo`` field set (call
        ``sdn_classifier.assign_silos()`` first).
        """
        # Group names by silo
        silo_names: dict[SDNSilo, list[tuple[str, SDNEntity]]] = defaultdict(list)
        for entity in sdn_list:
            for alias in entity.all_names:
                silo_names[entity.silo].append((alias, entity))

        if not silo_names:
            logger.warning("build_index called with empty SDN list.")
            return

        # For TF-IDF, fit the vectorizer on the entire corpus first so IDF
        # weights are global across all silos.
        if isinstance(self._embedder, _TFIDFEmbedder):
            all_names = [n for pairs in silo_names.values() for n, _ in pairs]
            self._embedder.fit(all_names)

        for silo, pairs in silo_names.items():
            names = [n for n, _ in pairs]
            logger.info(
                "Building FAISS index for silo=%s  entries=%d aliases=%d",
                silo.value, len({e.uid for _, e in pairs}), len(names),
            )
            vecs = self._embedder.encode(names)   # (N, D) float32, L2-normalised
            d = vecs.shape[1]

            idx = faiss.IndexFlatIP(d)            # cosine sim = dot product on unit vecs
            idx.add(vecs)

            self._faiss_indexes[silo] = idx
            self._name_index[silo] = pairs
            self._embeddings[silo] = vecs

        total = sum(idx.ntotal for idx in self._faiss_indexes.values())
        logger.info("Semantic index built: %d total vectors across %d silos.", total, len(self._faiss_indexes))

    @property
    def is_built(self) -> bool:
        return bool(self._faiss_indexes)

    # ------------------------------------------------------------------
    # Distance helpers (post-FAISS, on top-K subset only)
    # ------------------------------------------------------------------

    def _distances(self, silo: SDNSilo, query_vec: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Compute L1 or L2 distance from query_vec to selected rows of silo embeddings."""
        corpus = self._embeddings[silo]
        subset = corpus[indices]                  # (k, D)
        if self.distance_metric == DistanceMetric.L1:
            return np.sum(np.abs(subset - query_vec), axis=1)
        else:
            diff = subset - query_vec
            return np.sqrt(np.sum(diff * diff, axis=1))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        silos_to_search: list[SDNSilo],
        top_k: int = 10,
        threshold: float = 0.50,
        candidate_uids: Optional[set[str]] = None,
    ) -> list[SemanticCandidate]:
        """
        Search across the requested silos, deduplicate by UID, and return
        up to *top_k* candidates sorted by cosine similarity descending.

        Parameters
        ----------
        query_text:       Entity text from Comprehend.
        silos_to_search:  Silos to query (derived from Comprehend entity type).
        top_k:            Maximum results to return.
        threshold:        Minimum cosine similarity [0, 1] to keep.
        candidate_uids:   Optional set of SDN UIDs from fuzzy pre-filter;
                          if given, only these UIDs are kept from FAISS results.
        """
        if not self.is_built:
            raise RuntimeError("Call build_index() before querying.")

        query_vec = self._embedder.encode([query_text])[0]  # (D,)
        qv = query_vec.reshape(1, -1)

        all_hits: list[SemanticCandidate] = []

        for silo in silos_to_search:
            idx = self._faiss_indexes.get(silo)
            if idx is None or idx.ntotal == 0:
                continue

            k = min(top_k * 4, idx.ntotal)
            scores, indices = idx.search(qv, k)   # scores: (1, k)  indices: (1, k)
            scores, indices = scores[0], indices[0]

            # Compute L1/L2 distances for the retrieved subset
            valid_mask = indices >= 0
            valid_idx = indices[valid_mask]
            valid_scores = scores[valid_mask]

            if valid_idx.size == 0:
                continue

            dists = self._distances(silo, query_vec, valid_idx)

            for cosine, faiss_idx, dist in zip(valid_scores, valid_idx, dists):
                cosine = float(cosine)
                if cosine < threshold:
                    break  # FAISS returns sorted by score desc, so we can break early

                alias, entity = self._name_index[silo][faiss_idx]

                if candidate_uids is not None and entity.uid not in candidate_uids:
                    continue

                all_hits.append(
                    SemanticCandidate(
                        sdn_entity=entity,
                        cosine_similarity=cosine,
                        distance=float(dist),
                        matched_alias=alias,
                        semantic_tier=_assign_semantic_tier(cosine, threshold),
                        silo=silo,
                    )
                )

        # Deduplicate: keep best cosine per UID, then sort
        best_by_uid: dict[str, SemanticCandidate] = {}
        for hit in all_hits:
            uid = hit.sdn_entity.uid
            if uid not in best_by_uid or hit.cosine_similarity > best_by_uid[uid].cosine_similarity:
                best_by_uid[uid] = hit

        results = sorted(best_by_uid.values(), key=lambda c: c.cosine_similarity, reverse=True)
        return results[:top_k]
