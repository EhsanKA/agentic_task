"""Citation network analysis: graph construction, PageRank, and anomaly detection."""

from collections import defaultdict

import networkx as nx
import pandas as pd

from benchmark.config import PAGERANK_ALPHA, PAGERANK_MAX_ITER, PAGERANK_TOL


def analyze_citations(papers_df: pd.DataFrame, citations_df: pd.DataFrame) -> dict:
    """Build citation graph and detect anomalies.

    Returns dict with keys:
        citation_graph, in_degree, out_degree, orphan_citations, self_citations,
        temporal_anomalies, citation_ring_papers, pagerank_scores, top_cited_papers
    """
    valid_ids = set(papers_df["paper_id"].unique())
    paper_years = dict(zip(papers_df["paper_id"], papers_df["year"]))

    graph: dict[str, list[str]] = defaultdict(list)
    in_deg = {pid: 0 for pid in valid_ids}
    out_deg = {pid: 0 for pid in valid_ids}
    orphans: list[dict] = []
    self_cites: list[str] = []
    temporal_anomalies: list[dict] = []

    for _, row in citations_df.iterrows():
        src, dst = row["citing_paper"], row["cited_paper"]

        if src in valid_ids:
            graph[src].append(dst)
            out_deg[src] += 1
        if dst in valid_ids:
            in_deg[dst] += 1
        if dst not in valid_ids:
            orphans.append({"citing_paper": src, "cited_paper": dst})
        if src == dst:
            self_cites.append(src)

        # Temporal anomaly: citing paper is older than cited paper
        if src in paper_years and dst in paper_years and paper_years[src] < paper_years[dst]:
            temporal_anomalies.append({
                "citing_paper": src,
                "cited_paper": dst,
                "citing_year": paper_years[src],
                "cited_year": paper_years[dst],
            })

    self_cites = list(set(self_cites))
    graph = dict(graph)

    # Build NetworkX digraph for PageRank and cycle detection
    G = nx.DiGraph()
    G.add_nodes_from(valid_ids)
    for src, targets in graph.items():
        for dst in targets:
            if dst in valid_ids:
                G.add_edge(src, dst)

    # Detect citation rings (cycles of length >= 3)
    try:
        ring_papers: set[str] = set()
        for cycle in nx.simple_cycles(G):
            if len(cycle) >= 3:
                ring_papers.update(cycle)
        citation_ring_papers = list(ring_papers)
    except Exception:
        citation_ring_papers = []

    pagerank = nx.pagerank(G, alpha=PAGERANK_ALPHA, max_iter=PAGERANK_MAX_ITER, tol=PAGERANK_TOL)
    top_cited = [p for p, _ in sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:10]]

    return {
        "citation_graph": graph,
        "in_degree": in_deg,
        "out_degree": out_deg,
        "orphan_citations": orphans,
        "self_citations": self_cites,
        "temporal_anomalies": temporal_anomalies,
        "citation_ring_papers": citation_ring_papers,
        "pagerank_scores": pagerank,
        "top_cited_papers": top_cited,
    }
