"""Data models for the OFAC fuzzy + semantic matching pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DistanceMetric(str, Enum):
    L1 = "l1"   # Manhattan
    L2 = "l2"   # Euclidean


class EntityType(str, Enum):
    INDIVIDUAL = "Individual"
    ENTITY = "Entity"
    VESSEL = "Vessel"
    AIRCRAFT = "Aircraft"
    UNKNOWN = "Unknown"


@dataclass
class SDNEntity:
    """A single entry from the OFAC Specially Designated Nationals list."""

    uid: str
    name: str
    entity_type: EntityType
    aliases: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    # Flattened list: primary name + all aliases used during matching
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
    """Score contribution from a single matching layer."""

    fuzzy_score: float        # rapidfuzz ratio [0, 100]
    semantic_score: float     # cosine similarity [0, 1]
    distance: float           # raw L1 or L2 vector distance
    matched_alias: str        # which SDN name string produced this score


@dataclass
class MatchResult:
    """Final ranked candidate pairing a Comprehend entity with an SDN entry."""

    comprehend_entity: ComprehendEntity
    sdn_entity: SDNEntity
    layer_score: LayerScore
    combined_score: float     # weighted fusion [0, 1]
    rank: int = 0

    def __repr__(self) -> str:
        return (
            f"MatchResult(query={self.comprehend_entity.text!r}, "
            f"sdn={self.sdn_entity.name!r}, "
            f"combined={self.combined_score:.4f}, "
            f"fuzzy={self.layer_score.fuzzy_score:.1f}, "
            f"semantic={self.layer_score.semantic_score:.4f})"
        )
