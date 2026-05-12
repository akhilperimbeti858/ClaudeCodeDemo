"""Data models for the OFAC fuzzy + semantic matching pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DistanceMetric(str, Enum):
    L1 = "l1"   # Manhattan
    L2 = "l2"   # Euclidean


class EntityType(str, Enum):
    INDIVIDUAL = "Individual"
    ENTITY = "Entity"
    VESSEL = "Vessel"
    AIRCRAFT = "Aircraft"
    UNKNOWN = "Unknown"


class SDNSilo(str, Enum):
    """Three logical buckets used to partition the SDN index."""
    OFAC_ORG = "ofac_org"   # Sanctioned organisations / vessels / aircraft
    OFAC_POI = "ofac_poi"   # Sanctioned persons of interest (individuals)
    FTO = "fto"             # Foreign terrorist organisations (SDGT-programme entities)


class FuzzyTier(int, Enum):
    """Priority tier assigned by the fuzzy layer — lower number = stronger match."""
    EXACT = 1          # Normalised exact string match
    JARO_WINKLER = 2   # High Jaro-Winkler similarity (>= configurable threshold)
    PARTIAL = 3        # Token-set / token-sort / WRatio partial match


class SemanticTier(int, Enum):
    """Priority tier assigned by the semantic layer — lower number = stronger match."""
    STRONG = 1    # cosine >= 0.95
    GOOD = 2      # cosine >= 0.80
    PARTIAL = 3   # cosine >= pipeline threshold


class EntityFlag(str, Enum):
    """Per-entity outcome flag set after all matching is complete."""
    CLEAN = "clean"
    # Comprehend likely extracted noise rather than a real named entity
    NON_ENTITY = "non_entity"
    # The extracted text is probably a fragment; a larger context window is
    # needed before a reliable SDN match can be made
    NEEDS_CONTEXT_EXPANSION = "needs_context_expansion"


# ---------------------------------------------------------------------------
# Core domain objects
# ---------------------------------------------------------------------------

@dataclass
class SDNEntity:
    """A single entry from the OFAC Specially Designated Nationals list."""

    uid: str
    name: str
    entity_type: EntityType
    aliases: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    silo: SDNSilo = SDNSilo.OFAC_ORG         # assigned by sdn_classifier
    all_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.all_names = [self.name] + self.aliases


@dataclass
class ComprehendEntity:
    """A single entity extracted by AWS Comprehend custom entity recognition."""

    text: str
    entity_type: str          # As returned by Comprehend (e.g. "PERSON", "ORGANIZATION")
    score: float              # Comprehend confidence [0, 1]
    begin_offset: int = 0
    end_offset: int = 0


@dataclass
class LayerScore:
    """Raw score contributions from both matching layers."""

    fuzzy_score: float        # rapidfuzz score [0, 100]
    fuzzy_tier: FuzzyTier     # tier that produced the best fuzzy score
    semantic_score: float     # cosine similarity [0, 1]
    semantic_tier: SemanticTier
    distance: float           # raw L1 or L2 vector distance
    matched_alias: str        # which SDN name string produced the best scores


@dataclass
class MatchResult:
    """Final ranked candidate pairing a Comprehend entity with an SDN entry."""

    comprehend_entity: ComprehendEntity
    sdn_entity: SDNEntity
    layer_score: LayerScore
    combined_score: float          # weighted fusion [0, 1]
    silo: SDNSilo                  # which SDN silo the match came from
    flag: EntityFlag = EntityFlag.CLEAN
    flag_reason: str = ""
    rank: int = 0

    def __repr__(self) -> str:
        return (
            f"MatchResult(query={self.comprehend_entity.text!r}, "
            f"sdn={self.sdn_entity.name!r}, "
            f"silo={self.silo.value}, "
            f"combined={self.combined_score:.4f}, "
            f"fuzzy_tier={self.layer_score.fuzzy_tier.name}, "
            f"semantic_tier={self.layer_score.semantic_tier.name}, "
            f"flag={self.flag.value})"
        )
