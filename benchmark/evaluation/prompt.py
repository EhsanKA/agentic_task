"""Benchmark prompt for Research Paper Entity Extraction and Citation Analysis."""

BENCHMARK_PROMPT = """
# Research Paper Entity Extraction and Citation Analysis Benchmark

## Scenario

You are a data scientist building an automated pipeline for analyzing research paper metadata. Your goal is to load data from files, extract structured information, resolve entity ambiguities (including challenging edge cases), detect anomalies in the citation network, and produce a comprehensive analytical report saved as an artifact.

You must decide for yourself how to decompose the task, which intermediate computations to perform, and in what order.

The data contains deliberately challenging edge cases that require careful reasoning.

---

## Context

You have access to a data directory containing three files. The variable `DATA_DIR` (str) is already defined and points to the directory.

### Input Files

1. **`papers_metadata.json`** — ~100 paper records as a JSON list:
```python
[
    {
        "paper_id": str,           # e.g., "paper_0001"
        "title": str,
        "authors": list[str],      # e.g., ["J. Smith", "Maria Garcia"]
        "institution": str | None, # e.g., "MIT" or "Stanford University"
        "abstract": str,           # may be empty ""
        "keywords": list[str],     # may be []
        "venue": str,              # e.g., "NeurIPS", "ICML", "NIPS" (names may vary)
        "year": int,
        "publication_date": str    # ISO format "YYYY-MM-DD"
    },
    ...
]
```

2. **`citations.csv`** — Citation relationships with columns:
   - `citing_paper`: str (paper_id of the citing paper)
   - `cited_paper`: str (paper_id of the cited paper)

3. **`author_affiliations.json`** — Reference data for entity resolution. Structure is a dict-of-dicts keyed by ID:
```python
{
    "authors": {
        "auth_001": {
            "canonical_name": str,
            "known_variations": list[str],
            "primary_institution": str  # institution ID, e.g., "inst_001"
        },
        # NOTE: Some authors share initials but are DIFFERENT people
    },
    "institutions": {
        "inst_001": {
            "canonical_name": str,
            "known_variations": list[str],
            "country": str
        },
    },
    "disambiguation_notes": list[dict],
    "venue_notes": list[str]
}
```

### Data Challenges

The data contains edge cases you must handle:

**Standard:**
- Author name variations: "John Smith", "J. Smith", "Smith, John"
- Institution name variations: "MIT", "Massachusetts Institute of Technology"
- Missing fields: empty abstract or keywords
- Orphan citations: references to non-existent paper_ids
- Self-citations

**Advanced:**

1. **Ambiguous authors**: Some authors share initials but are different people at different institutions.
   - "J. Smith" at MIT is NOT the same person as "J. Smith" at Oxford
   - You must use institution context to disambiguate

2. **Typos/OCR errors**: Some names contain typos.
   - "Jonh Smith" -> "John Smith"
   - "Massachusets Institute of Technology" -> "MIT"
   - Use fuzzy matching or edit distance

3. **Citation ring detection**: A subset of papers cite each other in a circular pattern.
   - Identify groups where A->B->C->D->E->A (and cross-citations within)

4. **Temporal anomalies**: Some citations violate temporal logic.
   - A paper from 2021 citing a paper from 2023 is impossible
   - Identify and flag these

5. **Venue disambiguation**: Venues appear with different names.
   - "NIPS" -> "NeurIPS" (renamed in 2018)
   - Normalize to canonical forms

6. **Conflicting affiliations**: Some papers list an author with an incorrect institution.
   - Cross-reference with author_affiliations.json to detect mismatches

---

## Required Variables

You are free to choose the order and decomposition, but your implementation must produce all of the following.

### Data Variables (loaded from files)

| Variable | Type | Description |
|----------|------|-------------|
| `papers_df` | `pd.DataFrame` | Loaded from `papers_metadata.json` |
| `citations_df` | `pd.DataFrame` | Loaded from `citations.csv` |
| `affiliations_data` | `dict` | Loaded from `author_affiliations.json` |

### Entity Variables

| Variable | Type | Description |
|----------|------|-------------|
| `extracted_authors` | `list[dict]` | Each dict: `name`, `paper_ids`, `name_variations` |
| `extracted_institutions` | `list[dict]` | Each dict: `name`, `paper_ids`, `name_variations` |
| `extracted_topics` | `dict[str, int]` | Topic -> frequency count |
| `methods_from_abstracts` | `list[str]` | Research methods found in abstracts |

### Resolution Variables

| Variable | Type | Description |
|----------|------|-------------|
| `author_resolution_map` | `dict[str, str]` | Variation -> canonical name |
| `institution_resolution_map` | `dict[str, str]` | Variation -> canonical name |
| `resolved_author_count` | `int` | Unique authors after resolution |
| `resolved_institution_count` | `int` | Unique institutions after resolution |

### Citation Network Variables

| Variable | Type | Description |
|----------|------|-------------|
| `citation_graph` | `dict[str, list[str]]` | Adjacency list |
| `in_degree` | `dict[str, int]` | Incoming citations per paper |
| `out_degree` | `dict[str, int]` | Outgoing citations per paper |
| `pagerank_scores` | `dict[str, float]` | PageRank centrality scores |
| `top_cited_papers` | `list[str]` | Top 10 most cited paper_ids |
| `orphan_citations` | `list[dict]` | Citations to non-existent papers |
| `self_citations` | `list[str]` | Paper_ids that cite themselves |

### Anomaly Detection Variables

| Variable | Type | Description |
|----------|------|-------------|
| `citation_ring_papers` | `list[str]` | Paper_ids involved in citation rings |
| `temporal_anomalies` | `list[dict]` | Each dict: `citing_paper`, `cited_paper`, `citing_year`, `cited_year` |
| `ambiguous_author_resolutions` | `list[dict]` | Each dict: `name_variation`, `resolved_to`, `institution_used`, `reasoning` |
| `typo_corrections` | `list[dict]` | Each dict: `original`, `corrected`, `confidence` |
| `venue_normalizations` | `dict[str, str]` | Venue variation -> canonical name |
| `affiliation_conflicts` | `list[dict]` | Each dict: `paper_id`, `author`, `listed_institution`, `expected_institution` |

### Validation Variables

```python
validation_results: dict[str, bool]
```

Required keys:

| Key | What to Check |
|-----|---------------|
| `"papers_loaded_ok"` | papers_df has expected columns and >0 rows |
| `"citations_loaded_ok"` | citations_df has expected columns and >0 rows |
| `"affiliations_loaded_ok"` | affiliations_data is valid dict |
| `"no_duplicate_paper_ids"` | All paper_ids unique |
| `"authors_extracted"` | extracted_authors has >0 entries |
| `"institutions_extracted"` | extracted_institutions has >0 entries |
| `"resolution_maps_valid"` | Resolution maps are non-empty |
| `"citation_graph_built"` | citation_graph is non-empty |
| `"pagerank_computed"` | pagerank_scores is non-empty |
| `"orphans_identified"` | Checked for orphan citations |
| `"self_citations_identified"` | Checked for self-citations |
| `"all_pagerank_finite"` | All PageRank values are finite |
| `"citation_rings_checked"` | Checked for citation rings |
| `"temporal_anomalies_checked"` | Checked for temporal violations |
| `"ambiguous_authors_handled"` | Used institution context for disambiguation |
| `"typos_handled"` | Applied fuzzy matching for typos |
| `"venues_normalized"` | Normalized venue name variations |

### Summary Statistics

`summary_stats: dict[str, Any]` with required keys:

| Key | Type | Description |
|-----|------|-------------|
| `"total_papers"` | `int` | Total papers |
| `"total_citations"` | `int` | Total citation relationships |
| `"unique_authors_raw"` | `int` | Before resolution |
| `"unique_authors_resolved"` | `int` | After resolution |
| `"unique_institutions_raw"` | `int` | Before resolution |
| `"unique_institutions_resolved"` | `int` | After resolution |
| `"papers_with_missing_abstract"` | `int` | Empty/null abstract |
| `"papers_with_missing_keywords"` | `int` | Empty/null keywords |
| `"orphan_citation_count"` | `int` | Orphan citations |
| `"self_citation_count"` | `int` | Self-citations |
| `"avg_citations_per_paper"` | `float` | Average outgoing citations |
| `"most_common_venue"` | `str` | Most frequent venue |
| `"year_range"` | `tuple[int, int]` | (min_year, max_year) |
| `"citation_ring_count"` | `int` | Papers in citation rings |
| `"temporal_anomaly_count"` | `int` | Temporal violations |
| `"typo_correction_count"` | `int` | Typos corrected |
| `"affiliation_conflict_count"` | `int` | Affiliation mismatches |

### Final Report

```python
final_report: dict
```

Must have this structure:

```python
{
    "metadata": {
        "task": "Research Paper Entity Extraction and Citation Analysis",
        "papers_analyzed": int,
        "execution_timestamp": str
    },
    "entity_extraction": {
        "authors": {
            "total_unique": int,
            "top_5_by_paper_count": [{"name": str, "paper_count": int}, ...]
        },
        "institutions": {
            "total_unique": int,
            "top_5_by_paper_count": [{"name": str, "paper_count": int}, ...]
        },
        "topics": {
            "total_unique": int,
            "top_10_by_frequency": [{"topic": str, "count": int}, ...]
        }
    },
    "citation_analysis": {
        "total_citations": int,
        "top_10_cited_papers": [{"paper_id": str, "citation_count": int, "title": str}, ...],
        "orphan_citations": [{"citing_paper": str, "cited_paper": str}, ...],
        "self_citations": [str, ...],
        "network_statistics": {
            "avg_in_degree": float,
            "avg_out_degree": float,
            "max_in_degree": int,
            "max_out_degree": int
        }
    },
    "anomaly_detection": {
        "citation_rings": {
            "detected": bool,
            "papers_involved": [str, ...],
            "description": str
        },
        "temporal_anomalies": {
            "count": int,
            "examples": [{"citing": str, "cited": str, "issue": str}, ...]
        },
        "ambiguous_resolutions": [
            {"variation": str, "resolved_to": str, "method": str}, ...
        ],
        "typo_corrections": [
            {"original": str, "corrected": str}, ...
        ],
        "affiliation_conflicts": [
            {"paper_id": str, "author": str, "conflict": str}, ...
        ]
    },
    "data_quality": {
        "missing_abstracts": int,
        "missing_keywords": int,
        "missing_institutions": int,
        "duplicate_author_entries": int
    },
    "validation_summary": {
        "all_checks_passed": bool,
        "failed_checks": [str, ...]
    }
}
```

### Artifact Generation

You MUST save the final report to disk:

```python
import os
with open(os.path.join(DATA_DIR, "final_report.json"), "w") as f:
    json.dump(final_report, f, indent=2, default=str)
```

---

## Constraints

1. Load all data from files using DATA_DIR — do not assume data is pre-loaded
2. Do not hardcode specific paper IDs, author names, or institution names
3. Entity resolution must use institution context for disambiguation — "J. Smith" at MIT != "J. Smith" at Oxford
4. Typo handling must use fuzzy matching (e.g., Levenshtein distance)
5. PageRank with damping factor 0.85
6. Citation rings require cycle detection in the citation graph
7. Temporal anomalies require comparing publication years
8. All intermediate variables must be inspectable
9. Handle edge cases gracefully
10. Save final_report.json to DATA_DIR as an artifact

---

## Success Criteria

1. All validation checks pass
2. Data loaded from files (not hardcoded)
3. Entity resolution correctly disambiguates "J. Smith" at different institutions as different people
4. Citation rings are detected (at least one ring of 5 papers)
5. Temporal anomalies are detected (at least one)
6. Typos are corrected with fuzzy matching
7. Venue names are normalized (NIPS -> NeurIPS)
8. Affiliation conflicts are identified
9. PageRank scores sum to ~1.0
10. Final report follows the exact schema
11. final_report.json saved to disk
12. All numeric values are finite

---

## Required Headroom Variables

Your code MUST define these variables (tests will fail if missing):

```python
# Citation ring detection
citation_ring_papers: list[str] = [...]

# Temporal anomalies — citations where citing_year < cited_year
temporal_anomalies: list[dict] = [
    {'citing_paper': 'paper_X', 'cited_paper': 'paper_Y', 'citing_year': 2021, 'cited_year': 2023},
]

# Ambiguous author resolution
ambiguous_author_resolutions: list[dict] = [
    {'name_variation': 'J. Smith', 'resolved_to': 'John Smith', 'institution_used': 'MIT', 'reasoning': '...'},
]

# Typo corrections
typo_corrections: list[dict] = [
    {'original': 'Jonh Smith', 'corrected': 'John Smith', 'confidence': 0.9},
]
```

**Detection hints:**
- Citation rings: `networkx.simple_cycles()` on the citation graph
- Temporal anomalies: if citing_year < cited_year, it's anomalous
- Ambiguous authors: "J. Smith" at MIT vs "J. Smith" at Oxford are different people
- Typos: fuzzy matching (Levenshtein distance) — "Jonh Smith" -> "John Smith"

---

## Output Format

Your code must produce all required variables listed above. Wrap your solution in a single ```python ... ``` code block.
"""
