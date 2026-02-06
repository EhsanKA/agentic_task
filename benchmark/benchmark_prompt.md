# Research Paper Entity Extraction and Citation Analysis Benchmark

## Scenario

You are a data scientist tasked with building an automated pipeline for analyzing research paper metadata. Your goal is to extract structured information from a collection of research papers, resolve entity ambiguities, construct a citation network, and produce a comprehensive analytical report.

**You must decide for yourself how to decompose the task**, which intermediate computations to perform, and in what order. **Do not simply follow a fixed step-by-step structure.**

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
    "venue": str,              # e.g., "NeurIPS", "ICML"
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
        "auth_002": { ... },
        # ... more authors keyed by auth_XXX
    },
    "institutions": {
        "inst_001": {
            "canonical_name": str,       # e.g., "Massachusetts Institute of Technology"
            "known_variations": list[str], # e.g., ["MIT", "M.I.T."]
            "country": str
        },
        "inst_002": { ... },
        # ... more institutions keyed by inst_XXX
    }
}
```

### Data Challenges (Intentional)

The data contains edge cases you must handle:
- **Author name variations**: Same person appears as "John Smith", "J. Smith", "Smith, John"
- **Institution name variations**: Same institution appears as "MIT", "Massachusetts Institute of Technology"
- **Missing fields**: Some papers have empty abstract (`""`) or empty keywords (`[]`)
- **Orphan citations**: Some citations reference paper_ids that don't exist in papers_raw
- **Self-citations**: Some papers cite themselves

---

## Requirements

You are free to choose the order and decomposition of the task, but your final implementation must produce all of the following variables. **The groupings below are for documentation purposes only—they do not imply any particular execution order.**

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
| `extracted_authors` | `list[dict]` | List of extracted author entities. Each dict must have keys: `name` (str), `paper_ids` (list[str]), `name_variations` (list[str]) |
| `extracted_institutions` | `list[dict]` | List of extracted institution entities. Each dict must have keys: `name` (str), `paper_ids` (list[str]), `name_variations` (list[str]) |
| `extracted_topics` | `dict[str, int]` | Dictionary mapping topic/keyword strings to their frequency count across all papers |
| `methods_from_abstracts` | `list[str]` | List of research methods mentioned in abstracts (e.g., "gradient descent", "cross-validation", "attention mechanism") |

#### Resolution Variables

| Variable | Type | Description |
|----------|------|-------------|
| `author_resolution_map` | `dict[str, str]` | Maps each author name variation to its canonical form. Keys are variations, values are canonical names |
| `institution_resolution_map` | `dict[str, str]` | Maps each institution name variation to its canonical form |
| `resolved_author_count` | `int` | Number of unique authors after resolution |
| `resolved_institution_count` | `int` | Number of unique institutions after resolution |

#### Citation Network Variables

| Variable | Type | Description |
|----------|------|-------------|
| `citation_graph` | `dict[str, list[str]]` | Adjacency list where keys are paper_ids and values are lists of paper_ids they cite |
| `in_degree` | `dict[str, int]` | Number of times each paper is cited (incoming citations) |
| `out_degree` | `dict[str, int]` | Number of papers each paper cites (outgoing citations) |
| `pagerank_scores` | `dict[str, float]` | PageRank centrality score for each paper |
| `top_cited_papers` | `list[str]` | Top 10 most cited paper_ids (by in_degree), sorted descending |
| `orphan_citations` | `list[dict]` | List of citation records where cited_paper does not exist in papers_df. Each dict has keys: citing_paper, cited_paper |
| `self_citations` | `list[str]` | List of paper_ids that cite themselves |

#### Validation Variables

You must populate this dictionary:

```python
validation_results: dict[str, bool]
```

Required keys:

| Key | What to Check |
|-----|---------------|
| `"papers_loaded_ok"` | papers_df has expected columns and >0 rows |
| `"citations_loaded_ok"` | citations_df has expected columns and >0 rows |
| `"affiliations_loaded_ok"` | affiliations_data is a valid dict with 'authors' and 'institutions' keys |
| `"no_duplicate_paper_ids"` | All paper_ids in papers_df are unique |
| `"authors_extracted"` | extracted_authors has >0 entries |
| `"institutions_extracted"` | extracted_institutions has >0 entries |
| `"resolution_maps_valid"` | author_resolution_map and institution_resolution_map are non-empty dicts |
| `"citation_graph_built"` | citation_graph is a non-empty dict |
| `"pagerank_computed"` | pagerank_scores is a non-empty dict with float values |
| `"orphans_identified"` | orphan_citations check was performed (may be empty list) |
| `"self_citations_identified"` | self_citations check was performed (may be empty list) |
| `"all_pagerank_finite"` | All values in pagerank_scores are finite (no NaN/Inf) |

#### Summary Statistics

You must produce:

| Variable | Type | Description |
|----------|------|-------------|
| `summary_stats` | `dict[str, Any]` | Dictionary containing aggregated statistics |

Required keys in `summary_stats`:

| Key | Type | Description |
|-----|------|-------------|
| `"total_papers"` | `int` | Total number of papers |
| `"total_citations"` | `int` | Total number of citation relationships |
| `"unique_authors_raw"` | `int` | Unique author names before resolution |
| `"unique_authors_resolved"` | `int` | Unique authors after resolution |
| `"unique_institutions_raw"` | `int` | Unique institution names before resolution |
| `"unique_institutions_resolved"` | `int` | Unique institutions after resolution |
| `"papers_with_missing_abstract"` | `int` | Count of papers with empty/null abstract |
| `"papers_with_missing_keywords"` | `int` | Count of papers with empty/null keywords |
| `"orphan_citation_count"` | `int` | Number of orphan citations found |
| `"self_citation_count"` | `int` | Number of self-citations found |
| `"avg_citations_per_paper"` | `float` | Average number of outgoing citations per paper |
| `"most_common_venue"` | `str` | Most frequent publication venue |
| `"year_range"` | `tuple[int, int]` | (min_year, max_year) of publications |

#### Final Report

You must produce:

```python
final_report: dict
```

The `final_report` must have this exact structure:

```python
{
    "metadata": {
        "task": "Research Paper Entity Extraction and Citation Analysis",
        "papers_analyzed": int,
        "execution_timestamp": str  # ISO format
    },
    "entity_extraction": {
        "authors": {
            "total_unique": int,
            "top_5_by_paper_count": list[dict]  # [{"name": str, "paper_count": int}, ...]
        },
        "institutions": {
            "total_unique": int,
            "top_5_by_paper_count": list[dict]  # [{"name": str, "paper_count": int}, ...]
        },
        "topics": {
            "total_unique": int,
            "top_10_by_frequency": list[dict]  # [{"topic": str, "count": int}, ...]
        }
    },
    "citation_analysis": {
        "total_citations": int,
        "top_10_cited_papers": list[dict],  # [{"paper_id": str, "citation_count": int, "title": str}, ...]
        "orphan_citations": list[dict],
        "self_citations": list[str],
        "network_statistics": {
            "avg_in_degree": float,
            "avg_out_degree": float,
            "max_in_degree": int,
            "max_out_degree": int
        }
    },
    "data_quality": {
        "missing_abstracts": int,
        "missing_keywords": int,
        "missing_institutions": int,
        "duplicate_author_entries": int  # Papers where same author appears twice
    },
    "validation_summary": {
        "all_checks_passed": bool,
        "failed_checks": list[str]  # List of validation keys that failed
    }
}
```

---

## Constraints

1. **Do not hardcode any specific paper IDs, author names, or institution names** - your solution must work for any dataset following the same schema.

2. **Entity resolution must use fuzzy matching or the reference data** - do not assume exact string matches.

3. **PageRank must be computed using a proper implementation** - you may use networkx or implement the iterative algorithm with damping factor 0.85.

4. **All intermediate variables must be inspectable** - store results in the named variables, do not just compute and discard.

5. **Handle edge cases gracefully** - missing fields, empty lists, malformed data should not crash your pipeline.

---

## Output Format

Your final output must include:

1. All required variables populated and accessible
2. The `final_report` dictionary printed as formatted JSON
3. The `validation_results` dictionary showing all checks

```python
import json
print("=== VALIDATION RESULTS ===")
print(json.dumps(validation_results, indent=2))
print("\n=== FINAL REPORT ===")
print(json.dumps(final_report, indent=2))
```

---

## Success Criteria

Your solution is successful if:

1. **All validation checks pass** (`all(validation_results.values()) == True`)
2. **Entity resolution reduces author count** (`resolved_author_count < unique_authors_raw`)
3. **Orphan citations are correctly identified** (there is at least one)
4. **Self-citations are correctly identified** (there is at least one)
5. **PageRank scores sum to approximately 1.0** (±0.01)
6. **Final report follows the exact schema specified**
7. **All numeric values are finite** (no NaN, no Inf)
