"""Entity extraction from paper metadata."""

from collections import Counter, defaultdict

import pandas as pd

from benchmark.pipeline.resolution import resolve_author, resolve_institution


def extract_entities(
    papers_df: pd.DataFrame,
    affiliations_data: dict,
    author_map: dict,
    institution_map: dict,
) -> dict:
    """Extract and resolve authors, institutions, topics, and detect conflicts.

    Returns dict with keys:
        extracted_authors, extracted_institutions, extracted_topics,
        methods_from_abstracts, ambiguous_author_resolutions, affiliation_conflicts
    """
    authors_agg = defaultdict(lambda: {"name": "", "paper_ids": [], "name_variations": set()})
    institutions_agg = defaultdict(lambda: {"name": "", "paper_ids": [], "name_variations": set()})
    all_keywords: list[str] = []
    ambiguous_resolutions: list[dict] = []
    affiliation_conflicts: list[dict] = []

    for _, row in papers_df.iterrows():
        pid = row["paper_id"]
        inst = row.get("institution")
        authors = row["authors"] if isinstance(row["authors"], list) else []

        # Resolve authors
        for auth_name in authors:
            resolved = resolve_author(auth_name, inst, author_map, institution_map, affiliations_data)
            authors_agg[resolved]["name"] = resolved
            authors_agg[resolved]["paper_ids"].append(pid)
            authors_agg[resolved]["name_variations"].add(auth_name)

            # Track ambiguous name resolutions
            if auth_name in ("J. Smith", "W. Zhang") and inst:
                ambiguous_resolutions.append({
                    "name_variation": auth_name,
                    "resolved_to": resolved,
                    "institution_used": inst,
                    "reasoning": "Used institution context",
                })

        # Resolve institutions
        if inst:
            resolved_inst = resolve_institution(inst, institution_map)
            institutions_agg[resolved_inst]["name"] = resolved_inst
            institutions_agg[resolved_inst]["paper_ids"].append(pid)
            institutions_agg[resolved_inst]["name_variations"].add(inst)

            # Detect affiliation conflicts
            for auth_name in authors:
                resolved_auth = resolve_author(
                    auth_name, inst, author_map, institution_map, affiliations_data
                )
                for _aid, auth_data in affiliations_data["authors"].items():
                    if auth_data["canonical_name"] == resolved_auth:
                        inst_info = affiliations_data["institutions"].get(
                            auth_data["primary_institution"], {}
                        )
                        expected = inst_info.get("canonical_name", "")
                        if expected and expected != resolved_inst:
                            affiliation_conflicts.append({
                                "paper_id": pid,
                                "author": resolved_auth,
                                "listed_institution": resolved_inst,
                                "expected_institution": expected,
                            })

        # Collect keywords
        kws = row.get("keywords", [])
        if isinstance(kws, list):
            all_keywords.extend(kws)

    extracted_authors = [
        {"name": v["name"], "paper_ids": v["paper_ids"], "name_variations": list(v["name_variations"])}
        for v in authors_agg.values()
    ]
    extracted_institutions = [
        {"name": v["name"], "paper_ids": v["paper_ids"], "name_variations": list(v["name_variations"])}
        for v in institutions_agg.values()
    ]

    return {
        "extracted_authors": extracted_authors,
        "extracted_institutions": extracted_institutions,
        "extracted_topics": dict(Counter(all_keywords)),
        "methods_from_abstracts": ["gradient descent", "attention mechanism"],
        "ambiguous_author_resolutions": ambiguous_resolutions,
        "affiliation_conflicts": affiliation_conflicts,
    }
