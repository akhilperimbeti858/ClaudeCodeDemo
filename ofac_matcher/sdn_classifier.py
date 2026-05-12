"""
SDN silo classification and Comprehend-to-silo mapping.

Two responsibilities:
  1. classify_sdn_entity()  — assign each SDNEntity to one of three silos:
       OFAC_ORG  – sanctioned organisations, vessels, aircraft
       OFAC_POI  – sanctioned individuals (persons of interest)
       FTO       – foreign terrorist organisations (Entity entries in the
                   SDGT programme)

  2. comprehend_to_silos()  — map a Comprehend entity-type label to the
       list of SDN silos that should be searched.  This enforces the rule
       "only search OFAC ORG if Comprehend said ORGANIZATION" etc.
"""

from __future__ import annotations

from .models import EntityType, SDNEntity, SDNSilo

# Comprehend entity-type labels → SDN silos to search.
# Keys are upper-cased Comprehend type strings.
# Values are ordered lists: primary silo first.
_COMPREHEND_SILO_MAP: dict[str, list[SDNSilo]] = {
    # Standard Comprehend built-in types
    "ORGANIZATION":    [SDNSilo.OFAC_ORG, SDNSilo.FTO],
    "PERSON":          [SDNSilo.OFAC_POI],
    "LOCATION":        [SDNSilo.OFAC_ORG],
    "COMMERCIAL_ITEM": [SDNSilo.OFAC_ORG],
    "EVENT":           [SDNSilo.OFAC_ORG, SDNSilo.FTO],
    "TITLE":           [SDNSilo.OFAC_POI, SDNSilo.OFAC_ORG],
    "OTHER":           list(SDNSilo),
    # Custom Comprehend entity types a compliance team might define
    "OFAC_ENTITY":     [SDNSilo.OFAC_ORG, SDNSilo.FTO],
    "OFAC_PERSON":     [SDNSilo.OFAC_POI],
    "TERRORIST_ORG":   [SDNSilo.FTO],
    "TERRORIST":       [SDNSilo.OFAC_POI, SDNSilo.FTO],
    "VESSEL":          [SDNSilo.OFAC_ORG],
    "AIRCRAFT":        [SDNSilo.OFAC_ORG],
}

# OFAC programmes that designate an entity as a foreign terrorist organisation
_FTO_PROGRAMMES = frozenset({"SDGT", "SDNTK", "FTO"})


def classify_sdn_entity(entity: SDNEntity) -> SDNSilo:
    """
    Assign *entity* to exactly one SDNSilo.

    Rules (in priority order):
    1. Individuals → OFAC_POI
    2. Entities / Vessels / Aircraft with an FTO-related programme → FTO
    3. Everything else → OFAC_ORG
    """
    if entity.entity_type == EntityType.INDIVIDUAL:
        return SDNSilo.OFAC_POI

    if entity.entity_type in (EntityType.ENTITY, EntityType.VESSEL, EntityType.AIRCRAFT):
        if _FTO_PROGRAMMES.intersection(entity.programs):
            return SDNSilo.FTO

    return SDNSilo.OFAC_ORG


def comprehend_to_silos(comprehend_type: str) -> list[SDNSilo]:
    """
    Return the ordered list of SDN silos to search for a given Comprehend
    entity-type label.

    Falls back to all three silos for unknown / unmapped types so that no
    potential match is silently dropped.
    """
    return _COMPREHEND_SILO_MAP.get(comprehend_type.upper(), list(SDNSilo))


def assign_silos(sdn_list: list[SDNEntity]) -> None:
    """
    In-place update: set ``entity.silo`` for every entry in *sdn_list*.
    Call this once after loading the SDN list before building any index.
    """
    for entity in sdn_list:
        entity.silo = classify_sdn_entity(entity)
