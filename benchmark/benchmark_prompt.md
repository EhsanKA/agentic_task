# Research Paper Entity Extraction and Citation Analysis Benchmark
## (ENHANCED VERSION - Headroom Testing for Gemini 3 Pro)

## Scenario

You are a data scientist tasked with building an **advanced automated pipeline** for analyzing research paper metadata. Your goal is to extract structured information from a collection of research papers, resolve entity ambiguities (including challenging edge cases), detect anomalies in the citation network, and produce a comprehensive analytical report.

**You must decide for yourself how to decompose the task**, which intermediate computations to perform, and in what order. **Do not simply follow a fixed step-by-step structure.**

**This task contains deliberately challenging edge cases** that require careful reasoning to solve correctly.

---

## Context

You have access to three data sources (already loaded in memory):

### Input Data Structures

**`papers_raw`** (`list[dict]`): List of ~100 paper records. Each paper dict has this schema:
```python
{
    "paper_id": str,           # e.g., "paper_0001"
    "title": str,
    "authors": list[str],      # e.g., ["J. Smith", "Maria Garcia"]
    "institution": str | None, # e.g., "MIT" or "Stanford University"
    "abstract": str,           # May be empty string ""
    "keywords": list[str],     # e.g., ["machine learning", "neural networks"], may be []
    "venue": str,              # e.g., "NeurIPS", "ICML", "NIPS" (venue names may vary!)
    "year": int,
    "publication_date": str    # ISO format "YYYY-MM-DD"
}
```

**`citations_raw`** (`pd.DataFrame`): Citation relationships with columns:
- `citing_paper`: str (paper_id of the paper doing the citing)
- `cited_paper`: str (paper_id of the paper being cited)

**`affiliations_raw`** (`dict`): Reference data for entity resolution. **Structure is a dict-of-dicts keyed by ID**:
```python
{
    "authors": {
        "auth_001": {
            "canonical_name": str,       # e.g., "John Smith"
            "known_variations": list[str], # e.g., ["J. Smith", "John A. Smith"]
            "primary_institution": str   # Institution ID, e.g., "inst_001"
        },
        # ... more authors (NOTE: Some authors share initials but are DIFFERENT people!)
    },
    "institutions": {
        "inst_001": {
            "canonical_name": str,       # e.g., "Massachusetts Institute of Technology"
            "known_variations": list[str], # e.g., ["MIT", "M.I.T."]
            "country": str
        },
        # ... more institutions
    },
    "disambiguation_notes": list[dict],  # Hints about ambiguous entities
    "venue_notes": list[str]             # Notes about venue name changes
}
```

### Data Challenges (ENHANCED - Requires Advanced Reasoning)

The data contains **challenging edge cases** you must handle:

#### Basic Challenges (Standard)
- **Author name variations**: Same person appears as "John Smith", "J. Smith", "Smith, John"
- **Institution name variations**: Same institution appears as "MIT", "Massachusetts Institute of Technology"
- **Missing fields**: Some papers have empty abstract (`""`) or empty keywords (`[]`)
- **Orphan citations**: Some citations reference paper_ids that don't exist in papers_raw
- **Self-citations**: Some papers cite themselves

#### Advanced Challenges (Headroom Testing)

1. **⚠️ AMBIGUOUS AUTHORS**: Some authors share the same initials but are **DIFFERENT PEOPLE** at different institutions.
   - Example: "J. Smith" at MIT is **NOT** the same person as "J. Smith" at Oxford
   - You must use **institution context** to disambiguate
   - Naive merging will produce INCORRECT results

2. **⚠️ TYPOS/OCR ERRORS**: Some author and institution names contain typos.
   - Example: "Jonh Smith" should map to "John Smith"
   - Example: "Massachusets Institute of Technology" should map to "MIT"
   - You must use fuzzy matching or edit distance

3. **⚠️ CITATION RING DETECTION**: A subset of papers cite each other in a suspicious circular pattern.
   - Identify groups where: A→B→C→D→E→A (and cross-citations within)
   - These should be flagged as anomalous

4. **⚠️ TEMPORAL ANOMALIES**: Some citations violate temporal logic.
   - A paper from 2021 cannot cite a paper from 2023
   - Identify and flag these impossible citations

5. **⚠️ VENUE DISAMBIGUATION**: Venues may appear with different names.
   - "NIPS" → "NeurIPS" (renamed in 2018)
   - "CVPR" → "Conference on Computer Vision and Pattern Recognition"
   - These should be normalized to canonical forms

6. **⚠️ CONFLICTING AFFILIATIONS**: Some papers list an author with an incorrect institution.
   - Cross-reference with affiliations_raw to detect mismatches

---

## Requirements

You are free to choose the order and decomposition of the task, but your final implementation must produce all of the following variables.

### Required Variables

#### Data Variables

| Variable | Type | Description |
|----------|------|-------------|
| `papers_df` | `pd.DataFrame` | Papers data with columns: paper_id, title, authors, institution, abstract, keywords, venue, year, publication_date |
| `citations_df` | `pd.DataFrame` | Citation relationships with columns: citing_paper, cited_paper |
| `affiliations_data` | `dict` | Author affiliations reference data |

#### Entity Variables

| Variable | Type | Description |
|----------|------|-------------|
| `extracted_authors` | `list[dict]` | Each dict: `name`, `paper_ids`, `name_variations` |
| `extracted_institutions` | `list[dict]` | Each dict: `name`, `paper_ids`, `name_variations` |
| `extracted_topics` | `dict[str, int]` | Topic → frequency count |
| `methods_from_abstracts` | `list[str]` | Research methods found in abstracts |

#### Resolution Variables

| Variable | Type | Description |
|----------|------|-------------|
| `author_resolution_map` | `dict[str, str]` | Variation → canonical name |
| `institution_resolution_map` | `dict[str, str]` | Variation → canonical name |
| `resolved_author_count` | `int` | Unique authors after resolution |
| `resolved_institution_count` | `int` | Unique institutions after resolution |

#### Citation Network Variables

| Variable | Type | Description |
|----------|------|-------------|
| `citation_graph` | `dict[str, list[str]]` | Adjacency list |
| `in_degree` | `dict[str, int]` | Incoming citations per paper |
| `out_degree` | `dict[str, int]` | Outgoing citations per paper |
| `pagerank_scores` | `dict[str, float]` | PageRank centrality scores |
| `top_cited_papers` | `list[str]` | Top 10 most cited paper_ids |
| `orphan_citations` | `list[dict]` | Citations to non-existent papers |
| `self_citations` | `list[str]` | Paper_ids that cite themselves |

#### ⭐ NEW: Anomaly Detection Variables (Headroom)

| Variable | Type | Description |
|----------|------|-------------|
| `citation_ring_papers` | `list[str]` | Paper_ids involved in suspicious citation rings |
| `temporal_anomalies` | `list[dict]` | Citations where citing_paper.year > cited_paper.year. Each dict: `citing_paper`, `cited_paper`, `citing_year`, `cited_year` |
| `ambiguous_author_resolutions` | `list[dict]` | Cases where "J. Smith" was disambiguated. Each dict: `name_variation`, `resolved_to`, `institution_used`, `reasoning` |
| `typo_corrections` | `list[dict]` | Typos that were corrected. Each dict: `original`, `corrected`, `confidence` |
| `venue_normalizations` | `dict[str, str]` | Map of venue variations to canonical names |
| `affiliation_conflicts` | `list[dict]` | Papers where listed institution doesn't match author's known institution. Each dict: `paper_id`, `author`, `listed_institution`, `expected_institution` |

#### Validation Variables

```python
validation_results: dict[str, bool]
```

Required keys (ENHANCED):

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
| `"pagerank_computed"` | pagerank_scores is non-empty with floats |
| `"orphans_identified"` | Checked for orphan citations |
| `"self_citations_identified"` | Checked for self-citations |
| `"all_pagerank_finite"` | All PageRank values are finite |
| `"citation_rings_checked"` | ⭐ Checked for citation rings |
| `"temporal_anomalies_checked"` | ⭐ Checked for temporal violations |
| `"ambiguous_authors_handled"` | ⭐ Used institution context for disambiguation |
| `"typos_handled"` | ⭐ Applied fuzzy matching for typos |
| `"venues_normalized"` | ⭐ Normalized venue name variations |

#### Summary Statistics

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
| `"citation_ring_count"` | `int` | ⭐ Papers in citation rings |
| `"temporal_anomaly_count"` | `int` | ⭐ Temporal violations |
| `"typo_correction_count"` | `int` | ⭐ Typos corrected |
| `"affiliation_conflict_count"` | `int` | ⭐ Affiliation mismatches |

#### Final Report

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
    "anomaly_detection": {  # ⭐ NEW SECTION
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

---

## Constraints

1. **Do not hardcode specific paper IDs, author names, or institution names**
2. **Entity resolution MUST use institution context for disambiguation** - "J. Smith" at MIT ≠ "J. Smith" at Oxford
3. **Typo handling MUST use fuzzy matching** (e.g., Levenshtein distance)
4. **PageRank with damping factor 0.85**
5. **Citation rings require cycle detection** in the citation graph
6. **Temporal anomalies require comparing publication years**
7. **All intermediate variables must be inspectable**
8. **Handle edge cases gracefully**

---

## Success Criteria

Your solution is successful if:

1. **All validation checks pass** (including new headroom checks)
2. **Entity resolution correctly disambiguates** "J. Smith" at different institutions as different people
3. **Citation rings are detected** (there is at least one ring of 5 papers)
4. **Temporal anomalies are detected** (there is at least one)
5. **Typos are corrected** with fuzzy matching
6. **Venue names are normalized** (NIPS → NeurIPS)
7. **Affiliation conflicts are identified**
8. **PageRank scores sum to ~1.0**
9. **Final report follows the exact schema**
10. **All numeric values are finite**

---

## Output Format

```python
import json
print("=== VALIDATION RESULTS ===")
print(json.dumps(validation_results, indent=2))
print("\n=== FINAL REPORT ===")
print(json.dumps(final_report, indent=2, default=str))
```
