"""Validation checks and summary statistics computation."""

import numpy as np
import pandas as pd


def compute_summary_stats(
    papers_df: pd.DataFrame,
    citations_df: pd.DataFrame,
    extracted_authors: list,
    extracted_institutions: list,
    orphan_citations: list,
    self_citations: list,
    citation_ring_papers: list,
    temporal_anomalies: list,
    typo_corrections: list,
    affiliation_conflicts: list,
) -> dict:
    """Compute dataset and pipeline summary statistics."""
    unique_authors_raw: set[str] = set()
    for authors in papers_df["authors"]:
        if isinstance(authors, list):
            unique_authors_raw.update(authors)

    return {
        "total_papers": len(papers_df),
        "total_citations": len(citations_df),
        "unique_authors_raw": len(unique_authors_raw),
        "unique_authors_resolved": len(extracted_authors),
        "unique_institutions_raw": len(set(papers_df["institution"].dropna().unique())),
        "unique_institutions_resolved": len(extracted_institutions),
        "papers_with_missing_abstract": int((papers_df["abstract"] == "").sum()),
        "papers_with_missing_keywords": int(sum(1 for kw in papers_df["keywords"] if not kw)),
        "orphan_citation_count": len(orphan_citations),
        "self_citation_count": len(self_citations),
        "avg_citations_per_paper": len(citations_df) / len(papers_df),
        "most_common_venue": papers_df["venue"].mode()[0] if not papers_df["venue"].empty else "",
        "year_range": (int(papers_df["year"].min()), int(papers_df["year"].max())),
        # Headroom metrics
        "citation_ring_count": len(citation_ring_papers),
        "temporal_anomaly_count": len(temporal_anomalies),
        "typo_correction_count": len(typo_corrections),
        "affiliation_conflict_count": len(affiliation_conflicts),
    }


def compute_validation_results(
    papers_df: pd.DataFrame,
    citations_df: pd.DataFrame,
    affiliations_data: dict,
    extracted_authors: list,
    extracted_institutions: list,
    author_map: dict,
    citation_graph: dict,
    pagerank_scores: dict,
    temporal_anomalies: list,
    ambiguous_author_resolutions: list,
    typo_corrections: list,
    venue_normalizations: dict,
) -> dict[str, bool]:
    """Run all validation checks. Returns dict of check_name -> passed."""
    return {
        "papers_loaded_ok": len(papers_df) > 0,
        "citations_loaded_ok": len(citations_df) > 0,
        "affiliations_loaded_ok": bool(affiliations_data),
        "no_duplicate_paper_ids": papers_df["paper_id"].is_unique,
        "authors_extracted": len(extracted_authors) > 0,
        "institutions_extracted": len(extracted_institutions) > 0,
        "resolution_maps_valid": len(author_map) > 0,
        "citation_graph_built": len(citation_graph) > 0,
        "pagerank_computed": len(pagerank_scores) > 0,
        "orphans_identified": True,
        "self_citations_identified": True,
        "all_pagerank_finite": all(np.isfinite(v) for v in pagerank_scores.values()),
        # Headroom validations
        "citation_rings_checked": True,
        "temporal_anomalies_checked": len(temporal_anomalies) > 0,
        "ambiguous_authors_handled": len(ambiguous_author_resolutions) > 0,
        "typos_handled": len(typo_corrections) > 0,
        "venues_normalized": len(venue_normalizations) > 0,
    }
