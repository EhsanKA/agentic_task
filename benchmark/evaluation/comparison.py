"""Multi-run model comparison: trial runner, test collection, and reporting."""

import os
import time
import unittest
from typing import Any, Callable, Dict, List

import pandas as pd

from benchmark.evaluation.agent import execute_agent_code, extract_variables
from benchmark.evaluation.tests import ALL_TEST_CLASSES, set_context

# The four "headroom" tests that are designed to differentiate models.
HEADROOM_TESTS = [
    "test_citation_rings",
    "test_temporal_anomalies",
    "test_typo_corrections",
    "test_ambiguous_authors",
]


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_single_trial(
    context: str,
    model_name: str,
    data_dir: str,
    generate_fn: Callable[[str, str], str],
) -> Dict[str, Any]:
    """Run one agent trial: generate, execute, test, return results dict.

    Args:
        context:     The full prompt context string.
        model_name:  Model identifier (e.g. ``"google/gemini-3-pro-preview"``).
        data_dir:    Path to the benchmark data directory.
        generate_fn: ``fn(prompt, model_name) -> response_text``.
                     In Colab this is typically
                     ``lambda p, m: ai.generate_text(prompt=p, model_name=m)``.

    Returns:
        A dict with keys: model, tests_run, tests_passed, failures, errors,
        needed_retry, elapsed_sec, per_test (dict[str, bool]).
    """
    # Clean up any artifact from a previous run
    report_path = os.path.join(data_dir, "final_report.json")
    if os.path.exists(report_path):
        os.remove(report_path)

    t0 = time.time()
    agent_response = generate_fn(context, model_name)
    needed_retry = False

    exec_result = execute_agent_code(agent_response, data_dir)

    # Retry once on failure
    if exec_result and "__error__" in exec_result:
        needed_retry = True
        print(f"  First attempt failed: {exec_result['__error__']}")
        print("  Retrying with error context...")
        retry_prompt = (
            f"Your previous code produced this error:\n{exec_result['__traceback__']}\n\n"
            f"Fix the bug and return the corrected complete code.\n\n{context}"
        )
        agent_response = generate_fn(retry_prompt, model_name)
        exec_result = execute_agent_code(agent_response, data_dir)

    elapsed = time.time() - t0

    variables = extract_variables(exec_result) if exec_result else {}
    variables["_data_dir"] = data_dir

    test_outcomes = _collect_test_outcomes(variables)
    total = len(test_outcomes)
    passed = sum(test_outcomes.values())

    return {
        "model": model_name,
        "tests_run": total,
        "tests_passed": passed,
        "failures": total - passed,
        "errors": 0,
        "needed_retry": needed_retry,
        "elapsed_sec": round(elapsed, 1),
        "per_test": test_outcomes,
    }


# ---------------------------------------------------------------------------
# Full comparison loop
# ---------------------------------------------------------------------------

def run_comparison(
    models: List[str],
    num_runs: int,
    context: str,
    data_dir: str,
    generate_fn: Callable[[str, str], str],
) -> List[Dict[str, Any]]:
    """Run *num_runs* trials per model and return all results.

    Args:
        models:      List of model identifiers.
        num_runs:    Number of runs per model.
        context:     Full prompt context string.
        data_dir:    Path to data directory.
        generate_fn: ``fn(prompt, model_name) -> response_text``.

    Returns:
        A list of result dicts (one per trial), each augmented with a ``run``
        key (1-indexed).
    """
    all_results: List[Dict[str, Any]] = []

    for model_name in models:
        for run_idx in range(1, num_runs + 1):
            print(f"\n{'=' * 60}")
            print(f"  Model: {model_name}  |  Run {run_idx}/{num_runs}")
            print(f"{'=' * 60}")

            record = run_single_trial(context, model_name, data_dir, generate_fn)
            record["run"] = run_idx
            all_results.append(record)

            p, t = record["tests_passed"], record["tests_run"]
            print(f"  => {p}/{t} tests passed  |  retry={record['needed_retry']}  |  {record['elapsed_sec']:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"All runs complete: {len(all_results)} total")
    print(f"{'=' * 60}")
    return all_results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def pass_rate_table(all_results: List[dict], models: List[str]) -> pd.DataFrame:
    """Build a per-test pass-rate DataFrame across models.

    Columns are model names; rows are test names.
    Headroom tests are tagged with ``***`` in an extra column.
    """
    if not all_results:
        return pd.DataFrame()

    test_names = list(all_results[0]["per_test"].keys())
    rates: Dict[str, Dict[str, str]] = {}

    for model in models:
        model_runs = [r for r in all_results if r["model"] == model]
        n = len(model_runs)
        rates[model] = {
            t: f"{sum(1 for r in model_runs if r['per_test'].get(t, False))}/{n}"
            for t in test_names
        }

    df = pd.DataFrame(rates, index=test_names)
    df.index.name = "Test"
    df["headroom"] = df.index.map(lambda t: "***" if t in set(HEADROOM_TESTS) else "")
    return df


def summary_table(all_results: List[dict], models: List[str]) -> pd.DataFrame:
    """Build a high-level summary DataFrame (one row per model)."""
    rows = []
    for model in models:
        runs = [r for r in all_results if r["model"] == model]
        n = len(runs)
        headroom_passed = sum(
            1 for r in runs for ht in HEADROOM_TESTS if r["per_test"].get(ht, False)
        )
        rows.append({
            "Model": model.split("/")[-1],
            "Runs": n,
            "Avg passed": round(sum(r["tests_passed"] for r in runs) / n, 1),
            "Full-pass runs": sum(1 for r in runs if r["failures"] == 0 and r["errors"] == 0),
            "Headroom": f"{headroom_passed}/{n * len(HEADROOM_TESTS)}",
            "Retries": sum(1 for r in runs if r["needed_retry"]),
            "Avg time (s)": round(sum(r["elapsed_sec"] for r in runs) / n, 1),
        })
    return pd.DataFrame(rows)


def run_log_table(all_results: List[dict]) -> pd.DataFrame:
    """Build a per-run detail DataFrame."""
    return pd.DataFrame([
        {
            "Model": r["model"].split("/")[-1],
            "Run": r["run"],
            "Passed": r["tests_passed"],
            "Failed": r["failures"],
            "Errors": r["errors"],
            "Retry": r["needed_retry"],
            "Time (s)": r["elapsed_sec"],
        }
        for r in all_results
    ])


def print_verdict(all_results: List[dict], models: List[str]) -> None:
    """Print a final verdict comparing two models on headroom tests."""
    scores: Dict[str, Dict[str, float]] = {}
    for model in models:
        runs = [r for r in all_results if r["model"] == model]
        n = len(runs)
        avg_passed = sum(r["tests_passed"] for r in runs) / n
        headroom_rate = sum(
            1 for r in runs for ht in HEADROOM_TESTS if r["per_test"].get(ht, False)
        ) / (n * len(HEADROOM_TESTS))
        scores[model] = {"avg_passed": avg_passed, "headroom_rate": headroom_rate}

    print("=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    for m, s in scores.items():
        short = m.split("/")[-1]
        print(f"  {short:30s}  avg={s['avg_passed']:.1f}  headroom={s['headroom_rate']:.0%}")

    if len(scores) == 2:
        m0, m1 = list(scores.keys())
        s0, s1 = scores[m0], scores[m1]
        if s0["headroom_rate"] > s1["headroom_rate"]:
            winner, loser = m0, m1
        elif s1["headroom_rate"] > s0["headroom_rate"]:
            winner, loser = m1, m0
        else:
            winner = loser = None

        if winner and loser:
            ws, ls = scores[winner], scores[loser]
            print(f"\n  Winner: {winner.split('/')[-1]}")
            print(f"  Headroom advantage: {ws['headroom_rate']:.0%} vs {ls['headroom_rate']:.0%}")
            print(f"  Overall advantage:  {ws['avg_passed']:.1f} vs {ls['avg_passed']:.1f} avg tests passed")
        else:
            print("\n  Result: TIE on headroom tests")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_test_outcomes(variables: dict) -> Dict[str, bool]:
    """Run the full test suite silently and return {test_name: passed}."""
    set_context(variables)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for tc in ALL_TEST_CLASSES:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    result = unittest.TextTestRunner(verbosity=0).run(suite)

    failed = {str(case) for case, _ in result.failures + result.errors}

    outcomes: Dict[str, bool] = {}
    for tc in ALL_TEST_CLASSES:
        for method in loader.getTestCaseNames(tc):
            full = f"{method} ({tc.__module__}.{tc.__name__})"
            outcomes[method] = full not in failed
    return outcomes
