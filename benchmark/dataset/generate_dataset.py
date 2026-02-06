"""
Dataset Generation Script for Research Paper Entity Extraction Benchmark
(ENHANCED VERSION - With Headroom Challenges)

This script generates synthetic research paper data with:
- Known ground-truth entity mappings (for testing)
- Intentional variations/ambiguities to resolve
- Edge cases (missing fields, duplicates)

ENHANCED CHALLENGES (for Gemini 3 Pro headroom):
- Near-duplicate authors: Same initials, different people (J. Smith at MIT vs J. Smith at Oxford)
- Typos/OCR errors: "Jonh Smith", "John Smth" 
- Conflicting affiliations: Same author listed with different institutions
- Citation rings: Groups of papers that cite each other suspiciously
- Temporal anomalies: Citations to papers "from the future"
- Venue disambiguation: "NeurIPS" vs "NIPS" vs full name

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
# IMPORTANT: auth_001 and auth_011 are DIFFERENT people with similar names!
CANONICAL_AUTHORS = {
    "auth_001": {
        "canonical_name": "John Smith",
        "variations": ["J. Smith", "John A. Smith", "J. A. Smith", "Smith, John"],
        "typos": ["Jonh Smith", "John Smtih", "Jhon Smith"],  # OCR/typo errors
        "institution": "inst_001",
        "is_ambiguous_with": "auth_011"  # Different person with similar name
    },
    "auth_002": {
        "canonical_name": "Maria Garcia",
        "variations": ["M. Garcia", "Maria L. Garcia", "Garcia, Maria", "M. L. Garcia"],
        "typos": ["Maria Gracia", "Mara Garcia"],
        "institution": "inst_002",
        "is_ambiguous_with": None
    },
    "auth_003": {
        "canonical_name": "Wei Zhang",
        "variations": ["W. Zhang", "Wei W. Zhang", "Zhang, Wei", "Zhang Wei"],
        "typos": ["Wie Zhang", "Wei Zang"],
        "institution": "inst_003",
        "is_ambiguous_with": "auth_012"  # Different Wei Zhang at different institution
    },
    "auth_004": {
        "canonical_name": "Emily Johnson",
        "variations": ["E. Johnson", "Emily R. Johnson", "Johnson, Emily", "E. R. Johnson"],
        "typos": ["Emilly Johnson", "Emily Johsnon"],
        "institution": "inst_001",
        "is_ambiguous_with": None
    },
    "auth_005": {
        "canonical_name": "Ahmed Hassan",
        "variations": ["A. Hassan", "Ahmed M. Hassan", "Hassan, Ahmed", "A. M. Hassan"],
        "typos": ["Ahmd Hassan", "Ahmed Hasan"],
        "institution": "inst_004",
        "is_ambiguous_with": None
    },
    "auth_006": {
        "canonical_name": "Sarah Williams",
        "variations": ["S. Williams", "Sarah K. Williams", "Williams, Sarah", "S. K. Williams"],
        "typos": ["Sara Williams", "Sarah Willams"],
        "institution": "inst_002",
        "is_ambiguous_with": None
    },
    "auth_007": {
        "canonical_name": "Yuki Tanaka",
        "variations": ["Y. Tanaka", "Yuki S. Tanaka", "Tanaka, Yuki", "Tanaka Yuki"],
        "typos": ["Yuki Takana", "Yuki Tanka"],
        "institution": "inst_005",
        "is_ambiguous_with": None
    },
    "auth_008": {
        "canonical_name": "Michael Brown",
        "variations": ["M. Brown", "Michael J. Brown", "Brown, Michael", "M. J. Brown"],
        "typos": ["Micheal Brown", "Michael Browm"],
        "institution": "inst_003",
        "is_ambiguous_with": None
    },
    "auth_009": {
        "canonical_name": "Lisa Chen",
        "variations": ["L. Chen", "Lisa Y. Chen", "Chen, Lisa", "Chen Lisa"],
        "typos": ["Lisa Chem", "Lisa Chne"],
        "institution": "inst_004",
        "is_ambiguous_with": None
    },
    "auth_010": {
        "canonical_name": "David Miller",
        "variations": ["D. Miller", "David A. Miller", "Miller, David", "D. A. Miller"],
        "typos": ["Davd Miller", "David Miler"],
        "institution": "inst_005",
        "is_ambiguous_with": None
    },
    # DIFFERENT PERSON with same initials as auth_001 - THIS IS THE TRAP!
    "auth_011": {
        "canonical_name": "James Smith",  # Different first name, same last name
        "variations": ["J. Smith", "James B. Smith", "J. B. Smith", "Smith, James"],
        "typos": ["Jmaes Smith", "James Smtih"],
        "institution": "inst_004",  # DIFFERENT institution than auth_001
        "is_ambiguous_with": "auth_001",
        "disambiguation_hint": "Check institution - James Smith is at Oxford, John Smith is at MIT"
    },
    # DIFFERENT Wei Zhang at different institution
    "auth_012": {
        "canonical_name": "Wei Zhang",  # Same name, different person!
        "variations": ["W. Zhang", "Wei X. Zhang", "Zhang, Wei"],
        "typos": [],
        "institution": "inst_002",  # Stanford, not Tsinghua
        "is_ambiguous_with": "auth_003",
        "disambiguation_hint": "Same name but different middle initial and institution"
    }
}

# Canonical institutions with their variations
CANONICAL_INSTITUTIONS = {
    "inst_001": {
        "canonical_name": "Massachusetts Institute of Technology",
        "variations": ["MIT", "M.I.T.", "Massachusetts Inst. of Technology", "Mass. Institute of Technology"],
        "typos": ["Massachusets Institute of Technology", "MIT University"],
        "country": "USA"
    },
    "inst_002": {
        "canonical_name": "Stanford University",
        "variations": ["Stanford", "Stanford Univ.", "Stanford U.", "Leland Stanford Junior University"],
        "typos": ["Standford University", "Stanfrod University"],
        "country": "USA"
    },
    "inst_003": {
        "canonical_name": "Tsinghua University",
        "variations": ["Tsinghua", "Tsinghua Univ.", "Qinghua University", "THU"],
        "typos": ["Tsignhua University", "Tsingua University"],
        "country": "China"
    },
    "inst_004": {
        "canonical_name": "University of Oxford",
        "variations": ["Oxford", "Oxford Univ.", "Oxford University", "Univ. of Oxford"],
        "typos": ["Univeristy of Oxford", "Oxfrod University"],
        "country": "UK"
    },
    "inst_005": {
        "canonical_name": "University of Tokyo",
        "variations": ["Tokyo Univ.", "UTokyo", "Tokyo University", "Univ. of Tokyo"],
        "typos": ["Univeristy of Tokyo", "Tokoy University"],
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

# Publication venues WITH DISAMBIGUATION CHALLENGE
# Some venues changed names over time or have multiple forms
VENUES = {
    "neurips": {
        "canonical": "NeurIPS",
        "variations": ["NeurIPS", "NIPS", "Neural Information Processing Systems", "NeurIPS Conference"],
        "note": "NIPS was renamed to NeurIPS in 2018"
    },
    "icml": {
        "canonical": "ICML",
        "variations": ["ICML", "International Conference on Machine Learning"],
    },
    "iclr": {
        "canonical": "ICLR", 
        "variations": ["ICLR", "International Conference on Learning Representations"],
    },
    "aaai": {
        "canonical": "AAAI",
        "variations": ["AAAI", "AAAI Conference on Artificial Intelligence"],
    },
    "cvpr": {
        "canonical": "CVPR",
        "variations": ["CVPR", "IEEE/CVF CVPR", "Conference on Computer Vision and Pattern Recognition"],
    },
    "acl": {
        "canonical": "ACL",
        "variations": ["ACL", "Annual Meeting of the ACL", "Association for Computational Linguistics"],
    },
    "emnlp": {
        "canonical": "EMNLP",
        "variations": ["EMNLP", "Empirical Methods in Natural Language Processing"],
    },
    "kdd": {
        "canonical": "KDD",
        "variations": ["KDD", "ACM SIGKDD", "Knowledge Discovery and Data Mining"],
    }
}

# Citation ring members - papers that suspiciously cite each other
CITATION_RING_PAPERS = ["paper_0030", "paper_0031", "paper_0032", "paper_0033", "paper_0034"]

# Papers with temporal anomalies - will be cited before their publication date
TEMPORAL_ANOMALY_PAPERS = ["paper_0050", "paper_0051"]

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


def select_author_variation(author_id: str, use_typo: bool = False) -> str:
    """Select a random name variation for an author."""
    author = CANONICAL_AUTHORS[author_id]
    
    # 10% chance of typo if enabled and typos exist
    if use_typo and author.get("typos") and random.random() < 0.1:
        return random.choice(author["typos"])
    
    # 60% chance of using a variation, 40% canonical
    if random.random() > 0.4:
        return random.choice(author["variations"])
    return author["canonical_name"]


def select_institution_variation(inst_id: str, use_typo: bool = False) -> str:
    """Select a random name variation for an institution."""
    inst = CANONICAL_INSTITUTIONS[inst_id]
    
    # 5% chance of typo if enabled
    if use_typo and inst.get("typos") and random.random() < 0.05:
        return random.choice(inst["typos"])
    
    # 50% chance of using a variation
    if random.random() > 0.5:
        return random.choice(inst["variations"])
    return inst["canonical_name"]


def select_venue() -> str:
    """Select a random venue, sometimes using variations."""
    venue_key = random.choice(list(VENUES.keys()))
    venue_data = VENUES[venue_key]
    # 30% chance of using non-canonical variation
    if random.random() < 0.3:
        return random.choice(venue_data["variations"])
    return venue_data["canonical"]


def generate_papers(num_papers: int = 100) -> List[Dict[str, Any]]:
    """Generate synthetic paper metadata with ENHANCED edge cases for headroom testing."""
    papers = []
    # Include the ambiguous authors (auth_011, auth_012) in the pool
    author_ids = list(CANONICAL_AUTHORS.keys())
    
    base_date = datetime(2020, 1, 1)
    
    for i in range(num_papers):
        paper_id = generate_paper_id(i)
        
        # Select 1-4 authors randomly
        num_authors = random.randint(1, 4)
        selected_author_ids = random.sample(author_ids, num_authors)
        
        # Use name variations for authors (with occasional typos)
        use_typo = random.random() < 0.15  # 15% of papers have typos
        authors = [select_author_variation(aid, use_typo=use_typo) for aid in selected_author_ids]
        
        # Get institutions (with variations) for first author
        primary_inst_id = CANONICAL_AUTHORS[selected_author_ids[0]]["institution"]
        institution = select_institution_variation(primary_inst_id, use_typo=use_typo)
        
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
        
        # Publication info - use venue selection function
        venue = select_venue()
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
        
        # ========================================
        # BASIC EDGE CASES (from before)
        # ========================================
        if i == 5:  # Missing abstract
            paper["abstract"] = ""
        if i == 12:  # Missing keywords
            paper["keywords"] = []
        if i == 23:  # Very long author list
            extra_authors = random.sample([a for a in author_ids if a not in selected_author_ids], 3)
            paper["authors"].extend([select_author_variation(aid) for aid in extra_authors])
            paper["_ground_truth"]["author_ids"].extend(extra_authors)
        if i == 45:  # Missing institution
            paper["institution"] = None
        if i == 67:  # Duplicate author (same person listed twice with different name forms)
            dup_author = selected_author_ids[0]
            paper["authors"].append(CANONICAL_AUTHORS[dup_author]["variations"][0])
        
        # ========================================
        # ENHANCED EDGE CASES (for headroom)
        # ========================================
        
        # Case: Ambiguous "J. Smith" - ensure auth_001 and auth_011 both appear
        if i == 8:  # Paper by John Smith (MIT)
            paper["authors"] = ["J. Smith", select_author_variation("auth_002")]
            paper["institution"] = "MIT"
            paper["_ground_truth"]["author_ids"] = ["auth_001", "auth_002"]
            paper["_ground_truth"]["ambiguity_note"] = "J. Smith here is John Smith (MIT)"
            
        if i == 9:  # Paper by James Smith (Oxford) - DIFFERENT PERSON!
            paper["authors"] = ["J. Smith", select_author_variation("auth_005")]
            paper["institution"] = "Oxford"
            paper["_ground_truth"]["author_ids"] = ["auth_011", "auth_005"]
            paper["_ground_truth"]["ambiguity_note"] = "J. Smith here is James Smith (Oxford), NOT John Smith"
        
        # Case: Same name, different people (Wei Zhang at Tsinghua vs Wei Zhang at Stanford)
        if i == 18:  # Wei Zhang at Tsinghua
            paper["authors"] = ["Wei Zhang", select_author_variation("auth_006")]
            paper["institution"] = "Tsinghua University"
            paper["_ground_truth"]["author_ids"] = ["auth_003", "auth_006"]
            paper["_ground_truth"]["ambiguity_note"] = "Wei Zhang at Tsinghua (auth_003)"
            
        if i == 19:  # Wei Zhang at Stanford - DIFFERENT PERSON!
            paper["authors"] = ["W. Zhang", select_author_variation("auth_002")]
            paper["institution"] = "Stanford"
            paper["_ground_truth"]["author_ids"] = ["auth_012", "auth_002"]
            paper["_ground_truth"]["ambiguity_note"] = "W. Zhang at Stanford is auth_012, NOT auth_003"
        
        # Case: Author with CONFLICTING affiliations across papers
        if i == 25:  # Maria Garcia listed with wrong institution
            paper["authors"] = ["Maria Garcia"]
            paper["institution"] = "MIT"  # WRONG! She's at Stanford
            paper["_ground_truth"]["author_ids"] = ["auth_002"]
            paper["_ground_truth"]["conflicting_affiliation"] = True
            paper["_ground_truth"]["note"] = "Maria Garcia is at Stanford, not MIT - this is an error in the data"
        
        # Case: OCR/Typo errors that need fuzzy matching
        if i == 35:
            paper["authors"] = ["Jonh Smith", "Maria Gracia"]  # Typos!
            paper["institution"] = "Massachusets Institute of Technology"  # Typo!
            paper["_ground_truth"]["author_ids"] = ["auth_001", "auth_002"]
            paper["_ground_truth"]["has_typos"] = True
        
        # Case: Papers in citation ring (will cite each other)
        if paper_id in CITATION_RING_PAPERS:
            paper["_ground_truth"]["in_citation_ring"] = True
            paper["year"] = 2022  # All in same year
            paper["publication_date"] = "2022-06-15"
        
        # Case: Temporal anomaly papers (published in 2023, will be cited by 2021 paper)
        if paper_id in TEMPORAL_ANOMALY_PAPERS:
            paper["year"] = 2023
            paper["publication_date"] = "2023-01-15"
            paper["_ground_truth"]["is_temporal_anomaly_target"] = True
        
        # Case: Paper that cites future papers (temporal anomaly source)
        if i == 40:
            paper["year"] = 2021
            paper["publication_date"] = "2021-03-01"
            paper["_ground_truth"]["cites_future_papers"] = True
        
        # Case: Venue disambiguation challenge
        if i == 55:
            paper["venue"] = "NIPS"  # Old name
            paper["year"] = 2017
        if i == 56:
            paper["venue"] = "NeurIPS"  # New name
            paper["year"] = 2019
        if i == 57:
            paper["venue"] = "Neural Information Processing Systems"  # Full name
            paper["year"] = 2020
        
        papers.append(paper)
    
    return papers


def generate_citations(papers: List[Dict], density: float = 0.05) -> List[Dict[str, str]]:
    """Generate citation relationships with ENHANCED edge cases for headroom testing."""
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
    
    # ========================================
    # BASIC EDGE CASES (from before)
    # ========================================
    
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
    
    # ========================================
    # ENHANCED EDGE CASES (for headroom)
    # ========================================
    
    # CITATION RING: Papers that cite each other in a suspicious pattern
    # paper_0030 -> paper_0031 -> paper_0032 -> paper_0033 -> paper_0034 -> paper_0030
    ring = CITATION_RING_PAPERS
    for i in range(len(ring)):
        citing = ring[i]
        cited = ring[(i + 1) % len(ring)]  # Circular
        citations.append({
            "citing_paper": citing,
            "cited_paper": cited,
        })
    # Add some cross-citations within the ring for extra suspicion
    citations.append({"citing_paper": ring[0], "cited_paper": ring[2]})
    citations.append({"citing_paper": ring[1], "cited_paper": ring[3]})
    citations.append({"citing_paper": ring[2], "cited_paper": ring[4]})
    citations.append({"citing_paper": ring[3], "cited_paper": ring[0]})
    
    # TEMPORAL ANOMALY: Paper from 2021 cites papers from 2023 (impossible!)
    # paper_0040 (2021) cites paper_0050 and paper_0051 (2023)
    for future_paper in TEMPORAL_ANOMALY_PAPERS:
        citations.append({
            "citing_paper": "paper_0040",
            "cited_paper": future_paper,
        })
    
    # MULTIPLE SELF-CITATIONS from same paper (extra suspicious)
    citations.append({
        "citing_paper": "paper_0060",
        "cited_paper": "paper_0060"
    })
    citations.append({
        "citing_paper": "paper_0060",
        "cited_paper": "paper_0060"  # Duplicate self-citation!
    })
    
    return citations


def generate_author_affiliations() -> Dict[str, Any]:
    """Generate author-institution mapping data (with hints for disambiguation)."""
    affiliations = {
        "authors": {},
        "institutions": {},
        "disambiguation_notes": []  # Hints for the agent
    }
    
    # Add author records
    for auth_id, auth_data in CANONICAL_AUTHORS.items():
        affiliations["authors"][auth_id] = {
            "canonical_name": auth_data["canonical_name"],
            "known_variations": auth_data["variations"][:2],  # Only expose some variations
            "primary_institution": auth_data["institution"],
            "email_domain": CANONICAL_INSTITUTIONS[auth_data["institution"]]["canonical_name"].lower().replace(" ", "").replace("of", "")[:10] + ".edu"
        }
        
        # Add disambiguation hints for ambiguous authors
        if auth_data.get("is_ambiguous_with"):
            other_id = auth_data["is_ambiguous_with"]
            if other_id in CANONICAL_AUTHORS:
                other = CANONICAL_AUTHORS[other_id]
                affiliations["disambiguation_notes"].append({
                    "warning": f"'{auth_data['canonical_name']}' ({auth_id}) at {CANONICAL_INSTITUTIONS[auth_data['institution']]['canonical_name']} "
                              f"shares initials with '{other['canonical_name']}' ({other_id}) at {CANONICAL_INSTITUTIONS[other['institution']]['canonical_name']}. "
                              f"These are DIFFERENT people - use institution to disambiguate."
                })
    
    # Add institution records
    for inst_id, inst_data in CANONICAL_INSTITUTIONS.items():
        affiliations["institutions"][inst_id] = {
            "canonical_name": inst_data["canonical_name"],
            "known_variations": inst_data["variations"][:2],  # Only expose some variations
            "country": inst_data["country"]
        }
    
    # Add venue disambiguation notes
    affiliations["venue_notes"] = [
        "NIPS was renamed to NeurIPS in 2018. Papers before 2018 may use 'NIPS', papers after may use 'NeurIPS'. These should be treated as the same venue.",
        "Venues may appear as acronyms (CVPR) or full names (Conference on Computer Vision and Pattern Recognition)."
    ]
    
    return affiliations


def generate_ground_truth(papers: List[Dict]) -> Dict[str, Any]:
    """Generate ground truth data for unit testing (ENHANCED version)."""
    ground_truth = {
        "entity_resolution": {
            "authors": {},
            "institutions": {}
        },
        "expected_statistics": {},
        "validation_checks": {},
        "headroom_challenges": {}  # New section for advanced challenges
    }
    
    # Author resolution ground truth
    for auth_id, auth_data in CANONICAL_AUTHORS.items():
        all_vars = [auth_data["canonical_name"]] + auth_data["variations"]
        if auth_data.get("typos"):
            all_vars.extend(auth_data["typos"])
        ground_truth["entity_resolution"]["authors"][auth_data["canonical_name"]] = {
            "id": auth_id,
            "all_variations": all_vars,
            "is_ambiguous": auth_data.get("is_ambiguous_with") is not None
        }
    
    # Institution resolution ground truth
    for inst_id, inst_data in CANONICAL_INSTITUTIONS.items():
        all_vars = [inst_data["canonical_name"]] + inst_data["variations"]
        if inst_data.get("typos"):
            all_vars.extend(inst_data["typos"])
        ground_truth["entity_resolution"]["institutions"][inst_data["canonical_name"]] = {
            "id": inst_id,
            "all_variations": all_vars
        }
    
    # Expected statistics
    ground_truth["expected_statistics"] = {
        "total_papers": len(papers),
        "total_unique_authors": len(CANONICAL_AUTHORS),  # 12 including ambiguous ones
        "total_unique_institutions": len(CANONICAL_INSTITUTIONS),
        "papers_with_missing_abstract": 1,
        "papers_with_missing_keywords": 1,
        "papers_with_missing_institution": 1,
        "papers_with_duplicate_authors": 1,
        "papers_with_typos": 1,
        "papers_with_conflicting_affiliations": 1
    }
    
    # Validation checks that should pass
    ground_truth["validation_checks"] = {
        "all_paper_ids_unique": True,
        "citation_graph_has_orphans": True,
        "citation_graph_has_self_citations": True,
        "orphan_citation_paper_id": "paper_9999",
        "self_citation_paper_ids": ["paper_0015", "paper_0060"]
    }
    
    # HEADROOM CHALLENGES - Things that require advanced reasoning
    ground_truth["headroom_challenges"] = {
        "ambiguous_authors": {
            "description": "Authors with same initials but different identities",
            "cases": [
                {
                    "variation": "J. Smith",
                    "possible_identities": ["auth_001 (John Smith, MIT)", "auth_011 (James Smith, Oxford)"],
                    "disambiguation_method": "Use institution context"
                },
                {
                    "variation": "W. Zhang",
                    "possible_identities": ["auth_003 (Wei Zhang, Tsinghua)", "auth_012 (Wei Zhang, Stanford)"],
                    "disambiguation_method": "Use institution context and middle initial"
                }
            ]
        },
        "citation_ring": {
            "description": "Group of papers with suspicious mutual citation pattern",
            "papers": CITATION_RING_PAPERS,
            "expected_detection": "Should identify as anomalous due to circular citations"
        },
        "temporal_anomalies": {
            "description": "Citations that violate temporal logic (citing future papers)",
            "source_paper": "paper_0040",
            "cited_future_papers": TEMPORAL_ANOMALY_PAPERS,
            "expected_detection": "Should flag as impossible citations"
        },
        "typo_resolution": {
            "description": "Author/institution names with OCR or typing errors",
            "examples": [
                {"typo": "Jonh Smith", "canonical": "John Smith"},
                {"typo": "Maria Gracia", "canonical": "Maria Garcia"},
                {"typo": "Massachusets Institute of Technology", "canonical": "Massachusetts Institute of Technology"}
            ]
        },
        "venue_disambiguation": {
            "description": "Same venue appearing with different names",
            "examples": [
                {"variations": ["NeurIPS", "NIPS", "Neural Information Processing Systems"], "canonical": "NeurIPS"}
            ]
        },
        "conflicting_affiliations": {
            "description": "Author appearing with incorrect institution",
            "paper_id": "paper_0025",
            "author": "Maria Garcia",
            "listed_institution": "MIT",
            "correct_institution": "Stanford University"
        }
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
