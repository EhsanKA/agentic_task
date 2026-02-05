"""
Dataset Generation Script for Research Paper Entity Extraction Benchmark

This script generates synthetic research paper data with:
- Known ground-truth entity mappings (for testing)
- Intentional variations/ambiguities to resolve
- Edge cases (missing fields, duplicates)

Output files:
- papers_metadata.json: Main paper records
- citations.csv: Citation relationships
- author_affiliations.json: Author-institution mappings
- ground_truth.json: Expected entity resolutions for unit testing
"""

import json
import csv
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import hashlib

# Set seed for reproducibility
random.seed(42)

# ============================================================================
# GROUND TRUTH DATA - Canonical entities and their variations
# ============================================================================

# Canonical authors with their name variations
CANONICAL_AUTHORS = {
    "auth_001": {
        "canonical_name": "John Smith",
        "variations": ["J. Smith", "John A. Smith", "J. A. Smith", "Smith, John"],
        "institution": "inst_001"
    },
    "auth_002": {
        "canonical_name": "Maria Garcia",
        "variations": ["M. Garcia", "Maria L. Garcia", "Garcia, Maria", "M. L. Garcia"],
        "institution": "inst_002"
    },
    "auth_003": {
        "canonical_name": "Wei Zhang",
        "variations": ["W. Zhang", "Wei W. Zhang", "Zhang, Wei", "Zhang Wei"],
        "institution": "inst_003"
    },
    "auth_004": {
        "canonical_name": "Emily Johnson",
        "variations": ["E. Johnson", "Emily R. Johnson", "Johnson, Emily", "E. R. Johnson"],
        "institution": "inst_001"
    },
    "auth_005": {
        "canonical_name": "Ahmed Hassan",
        "variations": ["A. Hassan", "Ahmed M. Hassan", "Hassan, Ahmed", "A. M. Hassan"],
        "institution": "inst_004"
    },
    "auth_006": {
        "canonical_name": "Sarah Williams",
        "variations": ["S. Williams", "Sarah K. Williams", "Williams, Sarah", "S. K. Williams"],
        "institution": "inst_002"
    },
    "auth_007": {
        "canonical_name": "Yuki Tanaka",
        "variations": ["Y. Tanaka", "Yuki S. Tanaka", "Tanaka, Yuki", "Tanaka Yuki"],
        "institution": "inst_005"
    },
    "auth_008": {
        "canonical_name": "Michael Brown",
        "variations": ["M. Brown", "Michael J. Brown", "Brown, Michael", "M. J. Brown"],
        "institution": "inst_003"
    },
    "auth_009": {
        "canonical_name": "Lisa Chen",
        "variations": ["L. Chen", "Lisa Y. Chen", "Chen, Lisa", "Chen Lisa"],
        "institution": "inst_004"
    },
    "auth_010": {
        "canonical_name": "David Miller",
        "variations": ["D. Miller", "David A. Miller", "Miller, David", "D. A. Miller"],
        "institution": "inst_005"
    }
}

# Canonical institutions with their variations
CANONICAL_INSTITUTIONS = {
    "inst_001": {
        "canonical_name": "Massachusetts Institute of Technology",
        "variations": ["MIT", "M.I.T.", "Massachusetts Inst. of Technology", "Mass. Institute of Technology"],
        "country": "USA"
    },
    "inst_002": {
        "canonical_name": "Stanford University",
        "variations": ["Stanford", "Stanford Univ.", "Stanford U.", "Leland Stanford Junior University"],
        "country": "USA"
    },
    "inst_003": {
        "canonical_name": "Tsinghua University",
        "variations": ["Tsinghua", "Tsinghua Univ.", "Qinghua University", "THU"],
        "country": "China"
    },
    "inst_004": {
        "canonical_name": "University of Oxford",
        "variations": ["Oxford", "Oxford Univ.", "Oxford University", "Univ. of Oxford"],
        "country": "UK"
    },
    "inst_005": {
        "canonical_name": "University of Tokyo",
        "variations": ["Tokyo Univ.", "UTokyo", "Tokyo University", "Univ. of Tokyo"],
        "country": "Japan"
    }
}

# Research topics/keywords
RESEARCH_TOPICS = [
    "machine learning", "deep learning", "neural networks", "natural language processing",
    "computer vision", "reinforcement learning", "transformer models", "attention mechanisms",
    "graph neural networks", "federated learning", "transfer learning", "meta-learning",
    "generative models", "adversarial learning", "explainable AI", "optimization",
    "representation learning", "self-supervised learning", "multi-task learning", "few-shot learning"
]

# Research methods mentioned in abstracts
RESEARCH_METHODS = [
    "gradient descent", "backpropagation", "stochastic optimization", "cross-validation",
    "ablation study", "hyperparameter tuning", "ensemble methods", "regularization",
    "dropout", "batch normalization", "attention mechanism", "skip connections",
    "data augmentation", "pre-training", "fine-tuning", "knowledge distillation"
]

# Publication venues
VENUES = [
    "NeurIPS", "ICML", "ICLR", "AAAI", "CVPR", "ACL", "EMNLP", "NAACL",
    "ECCV", "ICCV", "KDD", "WWW", "SIGIR", "IJCAI", "UAI", "AISTATS"
]

# Abstract templates
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
    "We release our code and models for reproducibility."
]

# Fill-in values for templates
TEMPLATE_FILLS = {
    "method": ["DeepNet", "TransNet", "GraphFormer", "AttnNet", "MultiScale", "HierNet", "AdaptNet", 
               "RobustNet", "FastNet", "EfficientModel", "UnifiedNet", "FlexNet", "DynamicNet"],
    "benchmark": ["ImageNet", "COCO", "GLUE", "SQuAD", "WMT", "Citeseer", "PubMed", "MNIST", "CIFAR"],
    "challenge": ["scalability", "generalization", "data efficiency", "computational cost", 
                  "label noise", "distribution shift", "long-range dependencies"],
    "technique": ["self-attention", "graph convolution", "contrastive learning", "knowledge distillation",
                  "adversarial training", "curriculum learning", "multi-head attention"],
    "improvement": ["15", "23", "8", "31", "12", "19", "27", "6", "42"],
    "analysis": ["ablation studies", "sensitivity analysis", "error analysis", "qualitative analysis"],
    "old_approach": ["hand-crafted features", "fixed architectures", "single-scale processing", 
                    "supervised pre-training"],
    "aspect": ["semantic relationships", "hierarchical structure", "temporal dynamics", "spatial context"],
    "metric": ["accuracy", "F1 score", "BLEU score", "perplexity", "AUC", "mAP"],
    "property": ["efficient", "robust", "scalable", "interpretable", "flexible", "unified"],
    "component1": ["local attention", "global context", "residual connections", "layer normalization"],
    "component2": ["positional encoding", "gating mechanisms", "skip connections", "dropout"],
    "insight": ["sparse attention patterns", "hierarchical representations", "multi-scale features"],
    "capability": ["long-range modeling", "few-shot adaptation", "cross-domain transfer"],
    "advantage": ["fewer parameters", "less training data", "lower latency", "reduced memory"]
}


def generate_paper_id(index: int) -> str:
    """Generate a unique paper ID."""
    return f"paper_{index:04d}"


def generate_abstract(topics: List[str], methods: List[str]) -> str:
    """Generate a synthetic abstract using templates."""
    template = random.choice(ABSTRACT_TEMPLATES)
    
    # Fill in the template
    fills = {}
    for key, values in TEMPLATE_FILLS.items():
        fills[key] = random.choice(values)
    
    # Add actual topic and method mentions
    fills["topic"] = random.choice(topics)
    
    abstract = template.format(**fills)
    
    # Randomly inject additional method/topic mentions
    if random.random() > 0.5:
        method = random.choice(methods)
        abstract += f" We employ {method} in our implementation."
    
    return abstract


def select_author_variation(author_id: str) -> str:
    """Select a random name variation for an author."""
    author = CANONICAL_AUTHORS[author_id]
    # 60% chance of using a variation, 40% canonical
    if random.random() > 0.4:
        return random.choice(author["variations"])
    return author["canonical_name"]


def select_institution_variation(inst_id: str) -> str:
    """Select a random name variation for an institution."""
    inst = CANONICAL_INSTITUTIONS[inst_id]
    # 50% chance of using a variation
    if random.random() > 0.5:
        return random.choice(inst["variations"])
    return inst["canonical_name"]


def generate_papers(num_papers: int = 100) -> List[Dict[str, Any]]:
    """Generate synthetic paper metadata."""
    papers = []
    author_ids = list(CANONICAL_AUTHORS.keys())
    
    base_date = datetime(2020, 1, 1)
    
    for i in range(num_papers):
        paper_id = generate_paper_id(i)
        
        # Select 1-4 authors randomly
        num_authors = random.randint(1, 4)
        selected_author_ids = random.sample(author_ids, num_authors)
        
        # Use name variations for authors
        authors = [select_author_variation(aid) for aid in selected_author_ids]
        
        # Get institutions (with variations) for first author
        primary_inst_id = CANONICAL_AUTHORS[selected_author_ids[0]]["institution"]
        institution = select_institution_variation(primary_inst_id)
        
        # Select topics and methods
        paper_topics = random.sample(RESEARCH_TOPICS, random.randint(2, 4))
        paper_methods = random.sample(RESEARCH_METHODS, random.randint(1, 3))
        
        # Generate abstract
        abstract = generate_abstract(paper_topics, paper_methods)
        
        # Generate title
        main_topic = paper_topics[0].title()
        method_name = random.choice(TEMPLATE_FILLS["method"])
        title_templates = [
            f"{method_name}: A Novel Approach to {main_topic}",
            f"Improving {main_topic} with {method_name}",
            f"{method_name} for Efficient {main_topic}",
            f"Towards Better {main_topic}: The {method_name} Framework",
            f"Learning {main_topic} via {method_name}"
        ]
        title = random.choice(title_templates)
        
        # Publication info
        venue = random.choice(VENUES)
        pub_date = base_date + timedelta(days=random.randint(0, 1500))
        year = pub_date.year
        
        paper = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "institution": institution,
            "abstract": abstract,
            "keywords": paper_topics,
            "venue": venue,
            "year": year,
            "publication_date": pub_date.strftime("%Y-%m-%d"),
            "_ground_truth": {
                "author_ids": selected_author_ids,
                "institution_id": primary_inst_id,
                "methods_mentioned": paper_methods
            }
        }
        
        # Introduce some edge cases
        if i == 5:  # Missing abstract
            paper["abstract"] = ""
        if i == 12:  # Missing keywords
            paper["keywords"] = []
        if i == 23:  # Very long author list
            extra_authors = random.sample(author_ids, 3)
            paper["authors"].extend([select_author_variation(aid) for aid in extra_authors])
            paper["_ground_truth"]["author_ids"].extend(extra_authors)
        if i == 45:  # Missing institution
            paper["institution"] = None
        if i == 67:  # Duplicate author (same person listed twice with different name forms)
            dup_author = selected_author_ids[0]
            paper["authors"].append(CANONICAL_AUTHORS[dup_author]["variations"][0])
        
        papers.append(paper)
    
    return papers


def generate_citations(papers: List[Dict], density: float = 0.05) -> List[Dict[str, str]]:
    """Generate citation relationships between papers."""
    citations = []
    paper_ids = [p["paper_id"] for p in papers]
    paper_years = {p["paper_id"]: p["year"] for p in papers}
    
    for citing_paper in paper_ids:
        citing_year = paper_years[citing_paper]
        
        # A paper can only cite papers from the same or earlier years
        citable = [p for p in paper_ids if paper_years[p] <= citing_year and p != citing_paper]
        
        if citable:
            # Each paper cites some fraction of citable papers
            num_citations = max(1, int(len(citable) * density * random.uniform(0.5, 1.5)))
            num_citations = min(num_citations, len(citable), 10)  # Cap at 10 citations
            
            cited_papers = random.sample(citable, num_citations)
            for cited in cited_papers:
                citations.append({
                    "citing_paper": citing_paper,
                    "cited_paper": cited
                })
    
    # Add some edge cases
    # Orphan citation (citing non-existent paper)
    citations.append({
        "citing_paper": "paper_0010",
        "cited_paper": "paper_9999"  # Does not exist
    })
    
    # Self-citation (should be flagged)
    citations.append({
        "citing_paper": "paper_0015",
        "cited_paper": "paper_0015"
    })
    
    return citations


def generate_author_affiliations() -> Dict[str, Any]:
    """Generate author-institution mapping data."""
    affiliations = {
        "authors": {},
        "institutions": {}
    }
    
    # Add author records
    for auth_id, auth_data in CANONICAL_AUTHORS.items():
        affiliations["authors"][auth_id] = {
            "canonical_name": auth_data["canonical_name"],
            "known_variations": auth_data["variations"][:2],  # Only expose some variations
            "primary_institution": auth_data["institution"],
            "email_domain": CANONICAL_INSTITUTIONS[auth_data["institution"]]["canonical_name"].lower().replace(" ", "").replace("of", "")[:10] + ".edu"
        }
    
    # Add institution records
    for inst_id, inst_data in CANONICAL_INSTITUTIONS.items():
        affiliations["institutions"][inst_id] = {
            "canonical_name": inst_data["canonical_name"],
            "known_variations": inst_data["variations"][:2],  # Only expose some variations
            "country": inst_data["country"]
        }
    
    return affiliations


def generate_ground_truth(papers: List[Dict]) -> Dict[str, Any]:
    """Generate ground truth data for unit testing."""
    ground_truth = {
        "entity_resolution": {
            "authors": {},
            "institutions": {}
        },
        "expected_statistics": {},
        "validation_checks": {}
    }
    
    # Author resolution ground truth
    for auth_id, auth_data in CANONICAL_AUTHORS.items():
        ground_truth["entity_resolution"]["authors"][auth_data["canonical_name"]] = {
            "id": auth_id,
            "all_variations": [auth_data["canonical_name"]] + auth_data["variations"]
        }
    
    # Institution resolution ground truth
    for inst_id, inst_data in CANONICAL_INSTITUTIONS.items():
        ground_truth["entity_resolution"]["institutions"][inst_data["canonical_name"]] = {
            "id": inst_id,
            "all_variations": [inst_data["canonical_name"]] + inst_data["variations"]
        }
    
    # Expected statistics
    ground_truth["expected_statistics"] = {
        "total_papers": len(papers),
        "total_unique_authors": len(CANONICAL_AUTHORS),
        "total_unique_institutions": len(CANONICAL_INSTITUTIONS),
        "papers_with_missing_abstract": 1,
        "papers_with_missing_keywords": 1,
        "papers_with_missing_institution": 1,
        "papers_with_duplicate_authors": 1
    }
    
    # Validation checks that should pass
    ground_truth["validation_checks"] = {
        "all_paper_ids_unique": True,
        "citation_graph_has_orphans": True,  # We intentionally added one
        "citation_graph_has_self_citations": True,  # We intentionally added one
        "orphan_citation_paper_id": "paper_9999",
        "self_citation_paper_id": "paper_0015"
    }
    
    return ground_truth


def save_dataset(output_dir: str = "."):
    """Generate and save all dataset files."""
    print("Generating papers metadata...")
    papers = generate_papers(100)
    
    print("Generating citations...")
    citations = generate_citations(papers)
    
    print("Generating author affiliations...")
    affiliations = generate_author_affiliations()
    
    print("Generating ground truth...")
    ground_truth = generate_ground_truth(papers)
    
    # Remove ground truth from papers before saving (agent shouldn't see it)
    papers_clean = []
    for p in papers:
        paper_copy = {k: v for k, v in p.items() if k != "_ground_truth"}
        papers_clean.append(paper_copy)
    
    # Save papers_metadata.json
    papers_file = f"{output_dir}/papers_metadata.json"
    with open(papers_file, 'w') as f:
        json.dump(papers_clean, f, indent=2)
    print(f"Saved: {papers_file}")
    
    # Save citations.csv
    citations_file = f"{output_dir}/citations.csv"
    with open(citations_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["citing_paper", "cited_paper"])
        writer.writeheader()
        writer.writerows(citations)
    print(f"Saved: {citations_file}")
    
    # Save author_affiliations.json
    affiliations_file = f"{output_dir}/author_affiliations.json"
    with open(affiliations_file, 'w') as f:
        json.dump(affiliations, f, indent=2)
    print(f"Saved: {affiliations_file}")
    
    # Save ground_truth.json
    ground_truth_file = f"{output_dir}/ground_truth.json"
    with open(ground_truth_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved: {ground_truth_file}")
    
    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Papers: {len(papers_clean)}")
    print(f"Citations: {len(citations)}")
    print(f"Unique Authors: {len(CANONICAL_AUTHORS)}")
    print(f"Unique Institutions: {len(CANONICAL_INSTITUTIONS)}")
    print(f"Edge cases included: missing fields, orphan citations, self-citations, duplicate authors")


if __name__ == "__main__":
    save_dataset()
