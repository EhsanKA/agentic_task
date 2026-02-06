"""Data loading and dataset generation utilities."""

import json
import os
import tempfile

import pandas as pd

from benchmark.data.generate import save_dataset


def generate_data(output_dir: str | None = None) -> str:
    """Generate the synthetic dataset and return the output directory path."""
    if output_dir is None:
        output_dir = os.path.join(tempfile.gettempdir(), "benchmark_dataset")
    save_dataset(output_dir=output_dir)
    return output_dir


def load_data(data_dir: str) -> tuple:
    """Load generated dataset files.

    Returns:
        (papers_raw, citations_raw, affiliations_raw)
    """
    with open(os.path.join(data_dir, "papers_metadata.json")) as f:
        papers_raw = json.load(f)

    citations_raw = pd.read_csv(os.path.join(data_dir, "citations.csv"))

    with open(os.path.join(data_dir, "author_affiliations.json")) as f:
        affiliations_raw = json.load(f)

    print(f"Loaded {len(papers_raw)} papers, {len(citations_raw)} citations")
    return papers_raw, citations_raw, affiliations_raw


def load_ground_truth(data_dir: str) -> dict:
    """Load ground truth file if it exists."""
    path = os.path.join(data_dir, "ground_truth.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def setup_data(output_dir: str | None = None) -> tuple:
    """Generate and load data in one call.

    Returns:
        (papers_raw, citations_raw, affiliations_raw, data_dir)
    """
    data_dir = generate_data(output_dir)
    papers_raw, citations_raw, affiliations_raw = load_data(data_dir)
    return papers_raw, citations_raw, affiliations_raw, data_dir
