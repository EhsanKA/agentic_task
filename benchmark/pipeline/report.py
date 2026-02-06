"""Final report generation."""

from datetime import datetime

import numpy as np
import pandas as pd


def build_final_report(
    papers_df: pd.DataFrame,
    citations_df: pd.DataFrame,
    extracted_authors: list,
    extracted_institutions: list,
    extracted_topics: dict,
    in_degree: dict,
    out_degree: dict,
    top_cited_papers: list,
    orphan_citations: list,
    self_citations: list,
    citation_ring_papers: list,
    temporal_anomalies: list,
    ambiguous_author_resolutions: list,
    typo_corrections: list,
    affiliation_conflicts: list,
    summary_stats: dict,
    validation_results: dict,
) -> dict:
    """Build the structured final report."""
    author_counts = sorted(
        [(a["name"], len(a["paper_ids"])) for a in extracted_authors],
        key=lambda x: x[1], reverse=True,
    )
    inst_counts = sorted(
        [(i["name"], len(i["paper_ids"])) for i in extracted_institutions],
        key=lambda x: x[1], reverse=True,
    )
    top_topics = sorted(extracted_topics.items(), key=lambda x: x[1], reverse=True)[:10]

    top_cited_details = []
    for pid in top_cited_papers:
        row = papers_df[papers_df["paper_id"] == pid]
        if not row.empty:
            top_cited_details.append({
                "paper_id": pid,
                "citation_count": in_degree[pid],
                "title": row["title"].values[0],
            })

    return {
        "metadata": {
            "task": "Research Paper Entity Extraction and Citation Analysis",
            "papers_analyzed": len(papers_df),
            "execution_timestamp": datetime.now().isoformat(),
        },
        "entity_extraction": {
            "authors": {
                "total_unique": len(extracted_authors),
                "top_5_by_paper_count": [{"name": n, "paper_count": c} for n, c in author_counts[:5]],
            },
            "institutions": {
                "total_unique": len(extracted_institutions),
                "top_5_by_paper_count": [{"name": n, "paper_count": c} for n, c in inst_counts[:5]],
            },
            "topics": {
                "total_unique": len(extracted_topics),
                "top_10_by_frequency": [{"topic": t, "count": c} for t, c in top_topics],
            },
        },
        "citation_analysis": {
            "total_citations": len(citations_df),
            "top_10_cited_papers": top_cited_details,
            "orphan_citations": orphan_citations,
            "self_citations": self_citations,
            "network_statistics": {
                "avg_in_degree": float(np.mean(list(in_degree.values()))),
                "avg_out_degree": float(np.mean(list(out_degree.values()))),
                "max_in_degree": int(max(in_degree.values())),
                "max_out_degree": int(max(out_degree.values())),
            },
        },
        "anomaly_detection": {
            "citation_rings": {
                "detected": len(citation_ring_papers) > 0,
                "papers_involved": citation_ring_papers[:10],
                "description": "Papers with circular citation patterns",
            },
            "temporal_anomalies": {
                "count": len(temporal_anomalies),
                "examples": [
                    {"citing": t["citing_paper"], "cited": t["cited_paper"],
                     "issue": f"Year {t['citing_year']} cites {t['cited_year']}"}
                    for t in temporal_anomalies[:5]
                ],
            },
            "ambiguous_resolutions": ambiguous_author_resolutions[:5],
            "typo_corrections": typo_corrections,
            "affiliation_conflicts": [
                {"paper_id": c["paper_id"], "author": c["author"],
                 "conflict": f"Listed at {c['listed_institution']}, expected {c['expected_institution']}"}
                for c in affiliation_conflicts
            ],
        },
        "data_quality": {
            "missing_abstracts": summary_stats["papers_with_missing_abstract"],
            "missing_keywords": summary_stats["papers_with_missing_keywords"],
            "missing_institutions": int(papers_df["institution"].isna().sum()),
            "duplicate_author_entries": 0,
        },
        "validation_summary": {
            "all_checks_passed": all(validation_results.values()),
            "failed_checks": [k for k, v in validation_results.items() if not v],
        },
    }
