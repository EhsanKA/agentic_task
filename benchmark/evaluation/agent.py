"""Agent utilities: model selection, prompt construction, code extraction, and execution."""

import json
import os
import re
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from benchmark.config import DEFAULT_MODEL, FALLBACK_MODELS, REQUIRED_VARIABLES


def select_model(available_models: list, preferred: str = DEFAULT_MODEL) -> str:
    """Select the best available model with fallback chain."""
    if preferred in available_models:
        print(f"Selected model: {preferred}")
        return preferred

    print(f"'{preferred}' not available, trying fallbacks...")
    for fallback in FALLBACK_MODELS:
        if fallback in available_models:
            print(f"Using fallback: {fallback}")
            return fallback

    if available_models:
        print(f"Using first available: {available_models[0]}")
        return available_models[0]

    raise RuntimeError("No AI models available in Colab Pro")


def build_agent_context(prompt: str, data_dir: str) -> str:
    """Build the full prompt context with file paths and data samples."""
    papers_path = os.path.join(data_dir, "papers_metadata.json")
    citations_path = os.path.join(data_dir, "citations.csv")
    affiliations_path = os.path.join(data_dir, "author_affiliations.json")

    with open(papers_path) as f:
        sample_paper = json.load(f)[0]
    sample_citations = pd.read_csv(citations_path, nrows=3)
    with open(affiliations_path) as f:
        aff = json.load(f)

    return f"""
Write Python code to solve this task. You have one pre-defined variable:

- DATA_DIR = "{data_dir}"   (str, path to the data directory)

Files in DATA_DIR:
- papers_metadata.json  (~100 paper records)
- citations.csv         (citation relationships)
- author_affiliations.json (entity resolution reference data)

You MUST load these files yourself using DATA_DIR. Do NOT assume they are pre-loaded.

Pre-loaded modules and utilities available in the namespace:
  pd (pandas), np (numpy), json, re, os, nx (networkx),
  Counter (collections.Counter), defaultdict (collections.defaultdict),
  datetime (datetime.datetime)

Sample paper record:
{json.dumps(sample_paper, indent=2)}

Sample citations.csv:
{sample_citations.to_string()}

Sample affiliations structure:
- Keys: {list(aff.keys())}
- Sample author: {json.dumps(list(aff['authors'].values())[0], indent=2)}
- Sample institution: {json.dumps(list(aff['institutions'].values())[0], indent=2)}

TASK INSTRUCTIONS:
{prompt}

CRITICAL:
1. Load data from files using DATA_DIR — do not hardcode data
2. Save final_report.json to DATA_DIR when done
3. Return your solution as a single Python code block wrapped in ```python ... ```
"""


def extract_code_blocks(response_text: str) -> list[str]:
    """Extract Python code blocks from an LLM response."""
    patterns = [
        r"```python\n(.*?)```",
        r"```python\s*(.*?)```",
        r"```\n(.*?)```",
        r"```(.*?)```",
    ]
    for pattern in patterns:
        blocks = re.findall(pattern, response_text, re.DOTALL)
        if blocks:
            return blocks

    match = re.search(r"```python\s*\n?(import.*)", response_text, re.DOTALL)
    if match:
        return [match.group(1)]

    if response_text.strip().startswith("import"):
        return [response_text]

    return []


def execute_agent_code(response_text: str, data_dir: str) -> dict | None:
    """Extract code from agent response and execute it.

    Returns the exec globals dict on success, or dict with __error__ on failure.
    """
    code_blocks = extract_code_blocks(response_text)
    if not code_blocks:
        print("No code blocks found in response")
        print("Response preview:", response_text[:500])
        return None

    full_code = "\n\n".join(code_blocks)
    print(f"Extracted {len(code_blocks)} code block(s), executing...")

    exec_globals = {
        "DATA_DIR": data_dir,
        "pd": pd, "np": np, "json": json, "re": re, "os": os, "nx": nx,
        "defaultdict": defaultdict, "Counter": Counter, "datetime": datetime,
        "Dict": Dict, "List": List, "Any": Any, "Tuple": Tuple,
    }

    try:
        exec(full_code, exec_globals)
        print("Code executed successfully")
        return exec_globals
    except Exception as e:
        print(f"Error executing code: {e}")
        traceback.print_exc()
        return {"__error__": str(e), "__traceback__": traceback.format_exc()}


def extract_variables(exec_result: dict) -> dict:
    """Extract required output variables from execution result."""
    if "__error__" in exec_result:
        print(f"Cannot extract variables — execution failed: {exec_result['__error__']}")
        return {}

    extracted = {}
    missing = []
    for var in REQUIRED_VARIABLES:
        if var in exec_result:
            extracted[var] = exec_result[var]
        else:
            missing.append(var)

    print(f"Extracted {len(extracted)}/{len(REQUIRED_VARIABLES)} required variables")
    if missing:
        print(f"Missing: {missing}")
    return extracted
