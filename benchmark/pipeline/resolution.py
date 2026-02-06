"""Entity resolution: fuzzy matching, name normalization, and disambiguation."""

from benchmark.config import TYPO_MAP, VENUE_NORMALIZATION, FUZZY_MATCH_THRESHOLD

try:
    import Levenshtein
    _HAS_LEVENSHTEIN = True
except ImportError:
    _HAS_LEVENSHTEIN = False


def fuzzy_match(s1: str, s2: str, threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """Check if two strings are a fuzzy match using Levenshtein ratio."""
    if _HAS_LEVENSHTEIN:
        return Levenshtein.ratio(s1.lower(), s2.lower()) >= threshold
    return s1.lower() in s2.lower() or s2.lower() in s1.lower()


def build_resolution_maps(affiliations_data: dict) -> tuple[dict, dict, list]:
    """Build author and institution resolution maps from reference data.

    Returns:
        (author_map, institution_map, typo_corrections)
    """
    author_map: dict[str, str] = {}
    institution_map: dict[str, str] = {}
    typo_corrections: list[dict] = []

    for _aid, auth in affiliations_data["authors"].items():
        canonical = auth["canonical_name"]
        inst_id = auth["primary_institution"]
        author_map[canonical] = canonical
        for var in auth.get("known_variations", []):
            author_map[(var, inst_id)] = canonical
            author_map[var] = canonical

    for _iid, inst in affiliations_data["institutions"].items():
        canonical = inst["canonical_name"]
        institution_map[canonical] = canonical
        for var in inst.get("known_variations", []):
            institution_map[var] = canonical

    for typo, correct in TYPO_MAP.items():
        author_map[typo] = correct
        institution_map[typo] = correct
        typo_corrections.append({"original": typo, "corrected": correct, "confidence": 0.9})

    return author_map, institution_map, typo_corrections


def build_venue_normalizations() -> dict[str, str]:
    """Return venue name normalization map."""
    return dict(VENUE_NORMALIZATION)


def resolve_author(
    name: str,
    institution: str | None,
    author_map: dict,
    institution_map: dict,
    affiliations_data: dict,
) -> str:
    """Resolve an author name using institution context for disambiguation."""
    if institution:
        inst_canonical = institution_map.get(institution, institution)
        for _aid, auth in affiliations_data["authors"].items():
            primary = auth["primary_institution"]
            if name in auth["known_variations"] or name == auth["canonical_name"]:
                inst_info = affiliations_data["institutions"].get(primary, {})
                if (
                    primary == inst_canonical
                    or inst_info.get("canonical_name") == inst_canonical
                    or inst_canonical in inst_info.get("known_variations", [])
                ):
                    return auth["canonical_name"]
    return author_map.get(name, name)


def resolve_institution(name: str | None, institution_map: dict) -> str | None:
    """Resolve an institution name to its canonical form."""
    if not name:
        return name
    return institution_map.get(name, name)
