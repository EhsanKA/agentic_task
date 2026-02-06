"""Pipeline orchestrator: runs the full golden solution end-to-end."""

import pandas as pd

from benchmark.pipeline.resolution import build_resolution_maps, build_venue_normalizations
from benchmark.pipeline.extraction import extract_entities
from benchmark.pipeline.citations import analyze_citations
from benchmark.pipeline.validation import compute_summary_stats, compute_validation_results
from benchmark.pipeline.report import build_final_report


def run_pipeline(
    papers_raw: list,
    citations_raw: pd.DataFrame,
    affiliations_raw: dict,
) -> dict:
    """Execute the full entity extraction and citation analysis pipeline.

    Args:
        papers_raw: List of paper dicts from papers_metadata.json
        citations_raw: DataFrame of citation relationships
        affiliations_raw: Dict of author/institution reference data

    Returns:
        Dict containing all required output variables.
    """
    papers_df = pd.DataFrame(papers_raw)
    citations_df = citations_raw.copy()
    affiliations_data = affiliations_raw.copy()

    # Entity resolution
    author_map, institution_map, typo_corrections = build_resolution_maps(affiliations_data)
    venue_normalizations = build_venue_normalizations()

    # Entity extraction
    entities = extract_entities(papers_df, affiliations_data, author_map, institution_map)

    # Citation network analysis
    cites = analyze_citations(papers_df, citations_df)

    # Summary statistics
    summary_stats = compute_summary_stats(
        papers_df, citations_df,
        entities["extracted_authors"], entities["extracted_institutions"],
        cites["orphan_citations"], cites["self_citations"],
        cites["citation_ring_papers"], cites["temporal_anomalies"],
        typo_corrections, entities["affiliation_conflicts"],
    )

    # Validation
    validation_results = compute_validation_results(
        papers_df, citations_df, affiliations_data,
        entities["extracted_authors"], entities["extracted_institutions"],
        author_map, cites["citation_graph"], cites["pagerank_scores"],
        cites["temporal_anomalies"], entities["ambiguous_author_resolutions"],
        typo_corrections, venue_normalizations,
    )

    # Final report
    final_report = build_final_report(
        papers_df, citations_df,
        entities["extracted_authors"], entities["extracted_institutions"],
        entities["extracted_topics"],
        cites["in_degree"], cites["out_degree"], cites["top_cited_papers"],
        cites["orphan_citations"], cites["self_citations"],
        cites["citation_ring_papers"], cites["temporal_anomalies"],
        entities["ambiguous_author_resolutions"], typo_corrections,
        entities["affiliation_conflicts"], summary_stats, validation_results,
    )

    return {
        # Data
        "papers_df": papers_df,
        "citations_df": citations_df,
        "affiliations_data": affiliations_data,
        # Entities
        "extracted_authors": entities["extracted_authors"],
        "extracted_institutions": entities["extracted_institutions"],
        "extracted_topics": entities["extracted_topics"],
        "methods_from_abstracts": entities["methods_from_abstracts"],
        # Resolution
        "author_resolution_map": author_map,
        "institution_resolution_map": institution_map,
        "resolved_author_count": len(entities["extracted_authors"]),
        "resolved_institution_count": len(entities["extracted_institutions"]),
        # Citation network
        "citation_graph": cites["citation_graph"],
        "in_degree": cites["in_degree"],
        "out_degree": cites["out_degree"],
        "pagerank_scores": cites["pagerank_scores"],
        "top_cited_papers": cites["top_cited_papers"],
        "orphan_citations": cites["orphan_citations"],
        "self_citations": cites["self_citations"],
        # Anomaly detection
        "citation_ring_papers": cites["citation_ring_papers"],
        "temporal_anomalies": cites["temporal_anomalies"],
        "ambiguous_author_resolutions": entities["ambiguous_author_resolutions"],
        "typo_corrections": typo_corrections,
        "venue_normalizations": venue_normalizations,
        "affiliation_conflicts": entities["affiliation_conflicts"],
        # Validation & output
        "validation_results": validation_results,
        "summary_stats": summary_stats,
        "final_report": final_report,
    }
