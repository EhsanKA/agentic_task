"""Centralized configuration for the benchmark."""

# Model selection
DEFAULT_MODEL = "google/gemini-3-pro-preview"
FALLBACK_MODELS = [
    "google/gemini-3-pro-preview",
    "google/gemini-2.5-pro",
    "google/gemini-2.0-flash",
    "google/gemini-2.5-flash",
]

# PageRank parameters
PAGERANK_ALPHA = 0.85
PAGERANK_MAX_ITER = 100
PAGERANK_TOL = 1e-6

# Fuzzy matching
FUZZY_MATCH_THRESHOLD = 0.8

# Known typo corrections (author/institution names)
TYPO_MAP = {
    "Jonh Smith": "John Smith",
    "Maria Gracia": "Maria Garcia",
    "Massachusets Institute of Technology": "Massachusetts Institute of Technology",
    "Standford University": "Stanford University",
}

# Venue normalization (variation -> canonical)
VENUE_NORMALIZATION = {
    "NIPS": "NeurIPS",
    "Neural Information Processing Systems": "NeurIPS",
    "IEEE/CVF CVPR": "CVPR",
    "Annual Meeting of the ACL": "ACL",
    "International Conference on Machine Learning": "ICML",
}

# Required output variables (used by agent evaluation and tests)
REQUIRED_VARIABLES = [
    # Data
    "papers_df", "citations_df", "affiliations_data",
    # Entities
    "extracted_authors", "extracted_institutions", "extracted_topics", "methods_from_abstracts",
    # Resolution
    "author_resolution_map", "institution_resolution_map",
    "resolved_author_count", "resolved_institution_count",
    # Citation network
    "citation_graph", "in_degree", "out_degree", "pagerank_scores",
    "top_cited_papers", "orphan_citations", "self_citations",
    # Anomaly detection (headroom)
    "citation_ring_papers", "temporal_anomalies",
    "ambiguous_author_resolutions", "typo_corrections",
    "venue_normalizations", "affiliation_conflicts",
    # Validation & output
    "validation_results", "summary_stats", "final_report",
]
