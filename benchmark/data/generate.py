"""Synthetic dataset generation for the entity extraction benchmark.

Generates research paper metadata with intentional edge cases:
- Name variations and typos
- Ambiguous authors (same initials, different people)
- Citation rings and temporal anomalies
- Venue disambiguation challenges
"""

import csv
import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Canonical data definitions
# ---------------------------------------------------------------------------

CANONICAL_AUTHORS = {
    "auth_001": {
        "canonical_name": "John Smith",
        "variations": ["J. Smith", "John A. Smith", "J. A. Smith", "Smith, John"],
        "typos": ["Jonh Smith", "John Smtih", "Jhon Smith"],
        "institution": "inst_001",
        "is_ambiguous_with": "auth_011",
    },
    "auth_002": {
        "canonical_name": "Maria Garcia",
        "variations": ["M. Garcia", "Maria L. Garcia", "Garcia, Maria", "M. L. Garcia"],
        "typos": ["Maria Gracia", "Mara Garcia"],
        "institution": "inst_002",
        "is_ambiguous_with": None,
    },
    "auth_003": {
        "canonical_name": "Wei Zhang",
        "variations": ["W. Zhang", "Wei W. Zhang", "Zhang, Wei", "Zhang Wei"],
        "typos": ["Wie Zhang", "Wei Zang"],
        "institution": "inst_003",
        "is_ambiguous_with": "auth_012",
    },
    "auth_004": {
        "canonical_name": "Emily Johnson",
        "variations": ["E. Johnson", "Emily R. Johnson", "Johnson, Emily", "E. R. Johnson"],
        "typos": ["Emilly Johnson", "Emily Johsnon"],
        "institution": "inst_001",
        "is_ambiguous_with": None,
    },
    "auth_005": {
        "canonical_name": "Ahmed Hassan",
        "variations": ["A. Hassan", "Ahmed M. Hassan", "Hassan, Ahmed", "A. M. Hassan"],
        "typos": ["Ahmd Hassan", "Ahmed Hasan"],
        "institution": "inst_004",
        "is_ambiguous_with": None,
    },
    "auth_006": {
        "canonical_name": "Sarah Williams",
        "variations": ["S. Williams", "Sarah K. Williams", "Williams, Sarah", "S. K. Williams"],
        "typos": ["Sara Williams", "Sarah Willams"],
        "institution": "inst_002",
        "is_ambiguous_with": None,
    },
    "auth_007": {
        "canonical_name": "Yuki Tanaka",
        "variations": ["Y. Tanaka", "Yuki S. Tanaka", "Tanaka, Yuki", "Tanaka Yuki"],
        "typos": ["Yuki Takana", "Yuki Tanka"],
        "institution": "inst_005",
        "is_ambiguous_with": None,
    },
    "auth_008": {
        "canonical_name": "Michael Brown",
        "variations": ["M. Brown", "Michael J. Brown", "Brown, Michael", "M. J. Brown"],
        "typos": ["Micheal Brown", "Michael Browm"],
        "institution": "inst_003",
        "is_ambiguous_with": None,
    },
    "auth_009": {
        "canonical_name": "Lisa Chen",
        "variations": ["L. Chen", "Lisa Y. Chen", "Chen, Lisa", "Chen Lisa"],
        "typos": ["Lisa Chem", "Lisa Chne"],
        "institution": "inst_004",
        "is_ambiguous_with": None,
    },
    "auth_010": {
        "canonical_name": "David Miller",
        "variations": ["D. Miller", "David A. Miller", "Miller, David", "D. A. Miller"],
        "typos": ["Davd Miller", "David Miler"],
        "institution": "inst_005",
        "is_ambiguous_with": None,
    },
    # Trap: shares "J. Smith" with auth_001 but is a different person
    "auth_011": {
        "canonical_name": "James Smith",
        "variations": ["J. Smith", "James B. Smith", "J. B. Smith", "Smith, James"],
        "typos": ["Jmaes Smith", "James Smtih"],
        "institution": "inst_004",
        "is_ambiguous_with": "auth_001",
    },
    # Same full name as auth_003 at a different institution
    "auth_012": {
        "canonical_name": "Wei Zhang",
        "variations": ["W. Zhang", "Wei X. Zhang", "Zhang, Wei"],
        "typos": [],
        "institution": "inst_002",
        "is_ambiguous_with": "auth_003",
    },
}

CANONICAL_INSTITUTIONS = {
    "inst_001": {
        "canonical_name": "Massachusetts Institute of Technology",
        "variations": ["MIT", "M.I.T.", "Massachusetts Inst. of Technology", "Mass. Institute of Technology"],
        "typos": ["Massachusets Institute of Technology", "MIT University"],
        "country": "USA",
    },
    "inst_002": {
        "canonical_name": "Stanford University",
        "variations": ["Stanford", "Stanford Univ.", "Stanford U.", "Leland Stanford Junior University"],
        "typos": ["Standford University", "Stanfrod University"],
        "country": "USA",
    },
    "inst_003": {
        "canonical_name": "Tsinghua University",
        "variations": ["Tsinghua", "Tsinghua Univ.", "Qinghua University", "THU"],
        "typos": ["Tsignhua University", "Tsingua University"],
        "country": "China",
    },
    "inst_004": {
        "canonical_name": "University of Oxford",
        "variations": ["Oxford", "Oxford Univ.", "Oxford University", "Univ. of Oxford"],
        "typos": ["Univeristy of Oxford", "Oxfrod University"],
        "country": "UK",
    },
    "inst_005": {
        "canonical_name": "University of Tokyo",
        "variations": ["Tokyo Univ.", "UTokyo", "Tokyo University", "Univ. of Tokyo"],
        "typos": ["Univeristy of Tokyo", "Tokoy University"],
        "country": "Japan",
    },
}

RESEARCH_TOPICS = [
    "machine learning", "deep learning", "neural networks", "natural language processing",
    "computer vision", "reinforcement learning", "transformer models", "attention mechanisms",
    "graph neural networks", "federated learning", "transfer learning", "meta-learning",
    "generative models", "adversarial learning", "explainable AI", "optimization",
    "representation learning", "self-supervised learning", "multi-task learning", "few-shot learning",
]

RESEARCH_METHODS = [
    "gradient descent", "backpropagation", "stochastic optimization", "cross-validation",
    "ablation study", "hyperparameter tuning", "ensemble methods", "regularization",
    "dropout", "batch normalization", "attention mechanism", "skip connections",
    "data augmentation", "pre-training", "fine-tuning", "knowledge distillation",
]

VENUES = {
    "neurips": {"canonical": "NeurIPS", "variations": ["NeurIPS", "NIPS", "Neural Information Processing Systems", "NeurIPS Conference"]},
    "icml": {"canonical": "ICML", "variations": ["ICML", "International Conference on Machine Learning"]},
    "iclr": {"canonical": "ICLR", "variations": ["ICLR", "International Conference on Learning Representations"]},
    "aaai": {"canonical": "AAAI", "variations": ["AAAI", "AAAI Conference on Artificial Intelligence"]},
    "cvpr": {"canonical": "CVPR", "variations": ["CVPR", "IEEE/CVF CVPR", "Conference on Computer Vision and Pattern Recognition"]},
    "acl": {"canonical": "ACL", "variations": ["ACL", "Annual Meeting of the ACL", "Association for Computational Linguistics"]},
    "emnlp": {"canonical": "EMNLP", "variations": ["EMNLP", "Empirical Methods in Natural Language Processing"]},
    "kdd": {"canonical": "KDD", "variations": ["KDD", "ACM SIGKDD", "Knowledge Discovery and Data Mining"]},
}

CITATION_RING_PAPERS = ["paper_0030", "paper_0031", "paper_0032", "paper_0033", "paper_0034"]
TEMPORAL_ANOMALY_PAPERS = ["paper_0050", "paper_0051"]

ABSTRACT_TEMPLATES = [
    "We propose {method}, a novel approach to {topic} that achieves state-of-the-art results on {benchmark}. "
    "Our method leverages {technique} to address the challenge of {challenge}. "
    "Experiments demonstrate {improvement}% improvement over previous baselines. "
    "We conduct extensive {analysis} to validate our approach.",

    "This paper introduces {method} for {topic}. Unlike prior work that relies on {old_approach}, "
    "we utilize {technique} to capture {aspect}. Our approach is evaluated on {benchmark} "
    "and shows significant improvements in {metric}. We also provide theoretical analysis of {property}.",

    "Recent advances in {topic} have shown promising results using {technique}. "
    "However, existing methods struggle with {challenge}. We address this limitation by proposing {method}, "
    "which combines {component1} with {component2}. Comprehensive experiments on {benchmark} "
    "demonstrate the effectiveness of our approach.",

    "We present {method}, a {property} framework for {topic}. "
    "The key insight is that {insight} enables more effective {capability}. "
    "We validate our approach through experiments on {benchmark}, achieving {improvement}% gains. "
    "Ablation studies confirm the importance of {component1}.",

    "{topic} remains a challenging problem in machine learning. "
    "In this work, we propose {method} that addresses {challenge} through {technique}. "
    "Our model achieves competitive performance on {benchmark} while requiring {advantage}. "
    "We release our code and models for reproducibility.",
]

TEMPLATE_FILLS = {
    "method": ["DeepNet", "TransNet", "GraphFormer", "AttnNet", "MultiScale", "HierNet",
               "AdaptNet", "RobustNet", "FastNet", "EfficientModel", "UnifiedNet", "FlexNet", "DynamicNet"],
    "benchmark": ["ImageNet", "COCO", "GLUE", "SQuAD", "WMT", "Citeseer", "PubMed", "MNIST", "CIFAR"],
    "challenge": ["scalability", "generalization", "data efficiency", "computational cost",
                  "label noise", "distribution shift", "long-range dependencies"],
    "technique": ["self-attention", "graph convolution", "contrastive learning", "knowledge distillation",
                  "adversarial training", "curriculum learning", "multi-head attention"],
    "improvement": ["15", "23", "8", "31", "12", "19", "27", "6", "42"],
    "analysis": ["ablation studies", "sensitivity analysis", "error analysis", "qualitative analysis"],
    "old_approach": ["hand-crafted features", "fixed architectures", "single-scale processing", "supervised pre-training"],
    "aspect": ["semantic relationships", "hierarchical structure", "temporal dynamics", "spatial context"],
    "metric": ["accuracy", "F1 score", "BLEU score", "perplexity", "AUC", "mAP"],
    "property": ["efficient", "robust", "scalable", "interpretable", "flexible", "unified"],
    "component1": ["local attention", "global context", "residual connections", "layer normalization"],
    "component2": ["positional encoding", "gating mechanisms", "skip connections", "dropout"],
    "insight": ["sparse attention patterns", "hierarchical representations", "multi-scale features"],
    "capability": ["long-range modeling", "few-shot adaptation", "cross-domain transfer"],
    "advantage": ["fewer parameters", "less training data", "lower latency", "reduced memory"],
}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _generate_abstract(topics: List[str], methods: List[str]) -> str:
    template = random.choice(ABSTRACT_TEMPLATES)
    fills = {key: random.choice(values) for key, values in TEMPLATE_FILLS.items()}
    fills["topic"] = random.choice(topics)
    abstract = template.format(**fills)
    if random.random() > 0.5:
        abstract += f" We employ {random.choice(methods)} in our implementation."
    return abstract


def _select_author_variation(author_id: str, use_typo: bool = False) -> str:
    author = CANONICAL_AUTHORS[author_id]
    if use_typo and author.get("typos") and random.random() < 0.1:
        return random.choice(author["typos"])
    if random.random() > 0.4:
        return random.choice(author["variations"])
    return author["canonical_name"]


def _select_institution_variation(inst_id: str, use_typo: bool = False) -> str:
    inst = CANONICAL_INSTITUTIONS[inst_id]
    if use_typo and inst.get("typos") and random.random() < 0.05:
        return random.choice(inst["typos"])
    if random.random() > 0.5:
        return random.choice(inst["variations"])
    return inst["canonical_name"]


def _select_venue() -> str:
    venue_key = random.choice(list(VENUES.keys()))
    venue_data = VENUES[venue_key]
    if random.random() < 0.3:
        return random.choice(venue_data["variations"])
    return venue_data["canonical"]


def generate_papers(num_papers: int = 100) -> List[Dict[str, Any]]:
    """Generate paper metadata with edge cases for headroom testing."""
    papers = []
    author_ids = list(CANONICAL_AUTHORS.keys())
    base_date = datetime(2020, 1, 1)

    for i in range(num_papers):
        paper_id = f"paper_{i:04d}"
        num_authors = random.randint(1, 4)
        selected_ids = random.sample(author_ids, num_authors)
        use_typo = random.random() < 0.15

        authors = [_select_author_variation(aid, use_typo=use_typo) for aid in selected_ids]
        primary_inst_id = CANONICAL_AUTHORS[selected_ids[0]]["institution"]
        institution = _select_institution_variation(primary_inst_id, use_typo=use_typo)

        topics = random.sample(RESEARCH_TOPICS, random.randint(2, 4))
        methods = random.sample(RESEARCH_METHODS, random.randint(1, 3))

        method_name = random.choice(TEMPLATE_FILLS["method"])
        main_topic = topics[0].title()
        title = random.choice([
            f"{method_name}: A Novel Approach to {main_topic}",
            f"Improving {main_topic} with {method_name}",
            f"{method_name} for Efficient {main_topic}",
            f"Towards Better {main_topic}: The {method_name} Framework",
            f"Learning {main_topic} via {method_name}",
        ])

        venue = _select_venue()
        pub_date = base_date + timedelta(days=random.randint(0, 1500))

        paper = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "institution": institution,
            "abstract": _generate_abstract(topics, methods),
            "keywords": topics,
            "venue": venue,
            "year": pub_date.year,
            "publication_date": pub_date.strftime("%Y-%m-%d"),
            "_ground_truth": {
                "author_ids": selected_ids,
                "institution_id": primary_inst_id,
                "methods_mentioned": methods,
            },
        }

        # --- Standard edge cases ---
        if i == 5:
            paper["abstract"] = ""
        if i == 12:
            paper["keywords"] = []
        if i == 23:
            extra = random.sample([a for a in author_ids if a not in selected_ids], 3)
            paper["authors"].extend([_select_author_variation(aid) for aid in extra])
            paper["_ground_truth"]["author_ids"].extend(extra)
        if i == 45:
            paper["institution"] = None
        if i == 67:
            paper["authors"].append(CANONICAL_AUTHORS[selected_ids[0]]["variations"][0])

        # --- Headroom edge cases ---
        if i == 8:  # "J. Smith" at MIT = John Smith
            paper["authors"] = ["J. Smith", _select_author_variation("auth_002")]
            paper["institution"] = "MIT"
            paper["_ground_truth"]["author_ids"] = ["auth_001", "auth_002"]
        if i == 9:  # "J. Smith" at Oxford = James Smith (different person)
            paper["authors"] = ["J. Smith", _select_author_variation("auth_005")]
            paper["institution"] = "Oxford"
            paper["_ground_truth"]["author_ids"] = ["auth_011", "auth_005"]
        if i == 18:  # Wei Zhang at Tsinghua
            paper["authors"] = ["Wei Zhang", _select_author_variation("auth_006")]
            paper["institution"] = "Tsinghua University"
            paper["_ground_truth"]["author_ids"] = ["auth_003", "auth_006"]
        if i == 19:  # W. Zhang at Stanford (different person)
            paper["authors"] = ["W. Zhang", _select_author_variation("auth_002")]
            paper["institution"] = "Stanford"
            paper["_ground_truth"]["author_ids"] = ["auth_012", "auth_002"]
        if i == 25:  # Conflicting affiliation
            paper["authors"] = ["Maria Garcia"]
            paper["institution"] = "MIT"  # Wrong: she's at Stanford
            paper["_ground_truth"]["author_ids"] = ["auth_002"]
            paper["_ground_truth"]["conflicting_affiliation"] = True
        if i == 35:  # Typos
            paper["authors"] = ["Jonh Smith", "Maria Gracia"]
            paper["institution"] = "Massachusets Institute of Technology"
            paper["_ground_truth"]["author_ids"] = ["auth_001", "auth_002"]
            paper["_ground_truth"]["has_typos"] = True

        # Citation ring papers
        if paper_id in CITATION_RING_PAPERS:
            paper["year"] = 2022
            paper["publication_date"] = "2022-06-15"
            paper["_ground_truth"]["in_citation_ring"] = True

        # Temporal anomaly targets (future papers)
        if paper_id in TEMPORAL_ANOMALY_PAPERS:
            paper["year"] = 2023
            paper["publication_date"] = "2023-01-15"
            paper["_ground_truth"]["is_temporal_anomaly_target"] = True

        # Paper that cites future papers (temporal anomaly source)
        if i == 40:
            paper["year"] = 2021
            paper["publication_date"] = "2021-03-01"
            paper["_ground_truth"]["cites_future_papers"] = True

        # Venue disambiguation
        if i == 55:
            paper["venue"], paper["year"] = "NIPS", 2017
        if i == 56:
            paper["venue"], paper["year"] = "NeurIPS", 2019
        if i == 57:
            paper["venue"], paper["year"] = "Neural Information Processing Systems", 2020

        papers.append(paper)

    return papers


def generate_citations(papers: List[Dict], density: float = 0.05) -> List[Dict[str, str]]:
    """Generate citation relationships with edge cases."""
    citations: List[Dict[str, str]] = []
    paper_years = {p["paper_id"]: p["year"] for p in papers}
    paper_ids = list(paper_years.keys())

    for citing in paper_ids:
        citable = [p for p in paper_ids if paper_years[p] <= paper_years[citing] and p != citing]
        if citable:
            n = min(max(1, int(len(citable) * density * random.uniform(0.5, 1.5))), len(citable), 10)
            for cited in random.sample(citable, n):
                citations.append({"citing_paper": citing, "cited_paper": cited})

    # Orphan citation
    citations.append({"citing_paper": "paper_0010", "cited_paper": "paper_9999"})

    # Self-citations
    citations.append({"citing_paper": "paper_0015", "cited_paper": "paper_0015"})
    citations.append({"citing_paper": "paper_0060", "cited_paper": "paper_0060"})
    citations.append({"citing_paper": "paper_0060", "cited_paper": "paper_0060"})

    # Citation ring: circular pattern
    ring = CITATION_RING_PAPERS
    for j in range(len(ring)):
        citations.append({"citing_paper": ring[j], "cited_paper": ring[(j + 1) % len(ring)]})
    citations.append({"citing_paper": ring[0], "cited_paper": ring[2]})
    citations.append({"citing_paper": ring[1], "cited_paper": ring[3]})
    citations.append({"citing_paper": ring[2], "cited_paper": ring[4]})
    citations.append({"citing_paper": ring[3], "cited_paper": ring[0]})

    # Temporal anomaly: paper_0040 (2021) cites paper_0050, paper_0051 (2023)
    for future in TEMPORAL_ANOMALY_PAPERS:
        citations.append({"citing_paper": "paper_0040", "cited_paper": future})

    return citations


def generate_affiliations() -> Dict[str, Any]:
    """Generate author-institution reference data with disambiguation hints."""
    affiliations: Dict[str, Any] = {"authors": {}, "institutions": {}, "disambiguation_notes": []}

    for auth_id, auth_data in CANONICAL_AUTHORS.items():
        inst = CANONICAL_INSTITUTIONS[auth_data["institution"]]
        affiliations["authors"][auth_id] = {
            "canonical_name": auth_data["canonical_name"],
            "known_variations": auth_data["variations"][:2],
            "primary_institution": auth_data["institution"],
            "email_domain": inst["canonical_name"].lower().replace(" ", "").replace("of", "")[:10] + ".edu",
        }
        if auth_data.get("is_ambiguous_with") and auth_data["is_ambiguous_with"] in CANONICAL_AUTHORS:
            other = CANONICAL_AUTHORS[auth_data["is_ambiguous_with"]]
            other_inst = CANONICAL_INSTITUTIONS[other["institution"]]
            affiliations["disambiguation_notes"].append({
                "warning": (
                    f"'{auth_data['canonical_name']}' ({auth_id}) at {inst['canonical_name']} "
                    f"shares initials with '{other['canonical_name']}' ({auth_data['is_ambiguous_with']}) "
                    f"at {other_inst['canonical_name']}. These are DIFFERENT people."
                ),
            })

    for inst_id, inst_data in CANONICAL_INSTITUTIONS.items():
        affiliations["institutions"][inst_id] = {
            "canonical_name": inst_data["canonical_name"],
            "known_variations": inst_data["variations"][:2],
            "country": inst_data["country"],
        }

    affiliations["venue_notes"] = [
        "NIPS was renamed to NeurIPS in 2018.",
        "Venues may appear as acronyms or full names.",
    ]
    return affiliations


def generate_ground_truth(papers: List[Dict]) -> Dict[str, Any]:
    """Generate ground truth for unit testing."""
    gt: Dict[str, Any] = {
        "entity_resolution": {"authors": {}, "institutions": {}},
        "expected_statistics": {},
        "validation_checks": {},
        "headroom_challenges": {},
    }

    for auth_id, auth_data in CANONICAL_AUTHORS.items():
        all_vars = [auth_data["canonical_name"]] + auth_data["variations"] + auth_data.get("typos", [])
        gt["entity_resolution"]["authors"][auth_data["canonical_name"]] = {
            "id": auth_id, "all_variations": all_vars,
            "is_ambiguous": auth_data.get("is_ambiguous_with") is not None,
        }

    for inst_id, inst_data in CANONICAL_INSTITUTIONS.items():
        all_vars = [inst_data["canonical_name"]] + inst_data["variations"] + inst_data.get("typos", [])
        gt["entity_resolution"]["institutions"][inst_data["canonical_name"]] = {
            "id": inst_id, "all_variations": all_vars,
        }

    gt["expected_statistics"] = {
        "total_papers": len(papers),
        "total_unique_authors": len(CANONICAL_AUTHORS),
        "total_unique_institutions": len(CANONICAL_INSTITUTIONS),
    }
    gt["validation_checks"] = {
        "all_paper_ids_unique": True,
        "citation_graph_has_orphans": True,
        "self_citation_paper_ids": ["paper_0015", "paper_0060"],
    }
    gt["headroom_challenges"] = {
        "ambiguous_authors": [
            {"variation": "J. Smith", "identities": ["auth_001 (MIT)", "auth_011 (Oxford)"]},
            {"variation": "W. Zhang", "identities": ["auth_003 (Tsinghua)", "auth_012 (Stanford)"]},
        ],
        "citation_ring": {"papers": CITATION_RING_PAPERS},
        "temporal_anomalies": {"source": "paper_0040", "targets": TEMPORAL_ANOMALY_PAPERS},
        "typo_examples": [
            {"typo": "Jonh Smith", "canonical": "John Smith"},
            {"typo": "Maria Gracia", "canonical": "Maria Garcia"},
        ],
        "conflicting_affiliations": {
            "paper_id": "paper_0025", "author": "Maria Garcia",
            "listed": "MIT", "correct": "Stanford University",
        },
    }
    return gt


def save_dataset(output_dir: str = ".", seed: int = 42) -> None:
    """Generate and save all dataset files to output_dir."""
    random.seed(seed)

    papers = generate_papers(100)
    citations = generate_citations(papers)
    affiliations = generate_affiliations()
    ground_truth = generate_ground_truth(papers)

    papers_clean = [{k: v for k, v in p.items() if k != "_ground_truth"} for p in papers]

    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "papers_metadata.json"), "w") as f:
        json.dump(papers_clean, f, indent=2)

    with open(os.path.join(output_dir, "citations.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["citing_paper", "cited_paper"])
        writer.writeheader()
        writer.writerows(citations)

    with open(os.path.join(output_dir, "author_affiliations.json"), "w") as f:
        json.dump(affiliations, f, indent=2)

    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Dataset: {len(papers_clean)} papers, {len(citations)} citations, "
          f"{len(CANONICAL_AUTHORS)} authors, {len(CANONICAL_INSTITUTIONS)} institutions")
