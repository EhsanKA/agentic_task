"""Agent utilities: model selection, prompt construction, code extraction, and execution."""

import json
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


def build_agent_context(
    prompt: str,
    papers_raw: list,
    citations_raw: pd.DataFrame,
    affiliations_raw: dict,
) -> str:
    """Build the full prompt context with data samples for the agent."""
    return f"""
IMPORTANT: Write Python code to solve this task. The following variables are ALREADY LOADED in memory - DO NOT REDEFINE THEM:

- papers_raw: list[dict] with {len(papers_raw)} papers
- citations_raw: pd.DataFrame with {len(citations_raw)} citation rows
- affiliations_raw: dict with author/institution reference data

These variables exist and are ready to use. Reference them directly.

Sample paper from papers_raw:
{json.dumps(papers_raw[0], indent=2)}

Sample citations_raw columns: {citations_raw.columns.tolist()}
{citations_raw.head(3).to_string()}

Sample affiliations_raw structure:
- Keys: {list(affiliations_raw.keys())}
- Sample author entry: {json.dumps(list(affiliations_raw['authors'].values())[0], indent=2)}
- Sample institution entry: {json.dumps(list(affiliations_raw['institutions'].values())[0], indent=2)}

TASK INSTRUCTIONS:
{prompt}

CRITICAL:
1. DO NOT create fake/sample data - use the EXISTING papers_raw, citations_raw, affiliations_raw variables
2. Return your solution as a single Python code block wrapped in ```python ... ```
3. Your code will be executed with these variables already available in the namespace
"""


def extract_code_blocks(response_text: str) -> list[str]:
    """Extract Python code blocks from an LLM response.

    Tries multiple patterns to handle formatting variations.
    """
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

    # Unclosed code block
    match = re.search(r"```python\s*\n?(import.*)", response_text, re.DOTALL)
    if match:
        return [match.group(1)]

    # Raw Python code (no fences)
    if response_text.strip().startswith("import"):
        return [response_text]

    return []


def execute_agent_code(
    response_text: str,
    papers_raw: list,
    citations_raw: pd.DataFrame,
    affiliations_raw: dict,
) -> dict | None:
    """Extract code from agent response and execute it.

    Returns the exec globals dict on success, None on failure.
    """
    code_blocks = extract_code_blocks(response_text)
    if not code_blocks:
        print("No code blocks found in response")
        print("Response preview:", response_text[:500])
        return None

    full_code = "\n\n".join(code_blocks)
    print(f"Extracted {len(code_blocks)} code block(s), executing...")

    exec_globals = {
        "papers_raw": papers_raw,
        "citations_raw": citations_raw,
        "affiliations_raw": affiliations_raw,
        "pd": pd, "np": np, "json": json, "re": re, "nx": nx,
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
        return None


def extract_variables(exec_result: dict) -> dict:
    """Extract required output variables from execution result."""
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
