"""ofac_matcher — tiered fuzzy + FAISS semantic OFAC SDN entity matching pipeline."""

from .models import (
    ComprehendEntity,
    DistanceMetric,
    EntityFlag,
    EntityType,
    FuzzyTier,
    LayerScore,
    MatchResult,
    SDNEntity,
    SDNSilo,
    SemanticTier,
)
from .ofac_loader import load_sdn_from_dict, load_sdn_list
from .comprehend_parser import (
    from_raw_list,
    parse_async_job_output,
    parse_batch_response,
    parse_detect_entities_response,
)
from .sdn_classifier import assign_silos, classify_sdn_entity, comprehend_to_silos
from .entity_flagger import apply_flags, flag_entity
from .pipeline import OFACMatcher, PipelineConfig

__all__ = [
    "OFACMatcher",
    "PipelineConfig",
    "DistanceMetric",
    "EntityFlag",
    "EntityType",
    "FuzzyTier",
    "SemanticTier",
    "SDNSilo",
    "SDNEntity",
    "ComprehendEntity",
    "LayerScore",
    "MatchResult",
    "load_sdn_list",
    "load_sdn_from_dict",
    "assign_silos",
    "classify_sdn_entity",
    "comprehend_to_silos",
    "flag_entity",
    "apply_flags",
    "parse_detect_entities_response",
    "parse_batch_response",
    "parse_async_job_output",
    "from_raw_list",
]
