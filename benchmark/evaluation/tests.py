"""Unit tests for the Research Paper Entity Extraction benchmark.

Usage:
    from benchmark.evaluation.tests import set_context, run_all_tests
    set_context(solution_output_dict)
    run_all_tests()
"""

import unittest

import numpy as np
import pandas as pd

CTX: dict = {}


def set_context(variables: dict) -> None:
    """Set the test context with solution output variables."""
    global CTX
    CTX = variables


def _get(key: str):
    """Retrieve a variable from the test context."""
    if key not in CTX:
        raise KeyError(f"'{key}' not found in test context")
    return CTX[key]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestDataLoading(unittest.TestCase):

    def test_papers_df_not_empty(self):
        df = _get("papers_df")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_papers_df_columns(self):
        df = _get("papers_df")
        required = {"paper_id", "title", "authors", "institution", "abstract", "keywords", "venue", "year"}
        self.assertTrue(required.issubset(set(df.columns)))

    def test_citations_df_not_empty(self):
        df = _get("citations_df")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_affiliations_structure(self):
        aff = _get("affiliations_data")
        self.assertIsInstance(aff, dict)
        self.assertIn("authors", aff)


class TestEntityExtraction(unittest.TestCase):

    def test_authors_extracted(self):
        self.assertGreater(len(_get("extracted_authors")), 0)

    def test_institutions_extracted(self):
        self.assertGreater(len(_get("extracted_institutions")), 0)


class TestEntityResolution(unittest.TestCase):

    def test_author_map_populated(self):
        self.assertGreater(len(_get("author_resolution_map")), 0)

    def test_resolved_count_positive(self):
        self.assertGreater(_get("resolved_author_count"), 0)


class TestCitationNetwork(unittest.TestCase):

    def test_graph_not_empty(self):
        self.assertGreater(len(_get("citation_graph")), 0)

    def test_pagerank_sums_to_one(self):
        scores = _get("pagerank_scores")
        self.assertAlmostEqual(sum(scores.values()), 1.0, delta=0.01)

    def test_orphan_citations(self):
        self.assertGreater(len(_get("orphan_citations")), 0)

    def test_self_citations(self):
        self.assertGreater(len(_get("self_citations")), 0)


class TestHeadroomChallenges(unittest.TestCase):
    """Tests for advanced reasoning — weaker models tend to fail these."""

    def test_citation_rings(self):
        rings = _get("citation_ring_papers")
        self.assertIsInstance(rings, list)
        self.assertGreater(len(rings), 0, "Must detect citation rings")

    def test_temporal_anomalies(self):
        anomalies = _get("temporal_anomalies")
        self.assertIsInstance(anomalies, list)
        self.assertGreater(len(anomalies), 0, "Must detect temporal anomalies")

    def test_typo_corrections(self):
        corrections = _get("typo_corrections")
        self.assertIsInstance(corrections, list)
        self.assertGreater(len(corrections), 0, "Must correct typos")

    def test_ambiguous_authors(self):
        resolutions = _get("ambiguous_author_resolutions")
        self.assertIsInstance(resolutions, list)
        self.assertGreater(len(resolutions), 0, "Must disambiguate authors like J. Smith")


class TestValidationResults(unittest.TestCase):

    def test_is_dict(self):
        self.assertIsInstance(_get("validation_results"), dict)

    def test_headroom_keys_present(self):
        vr = _get("validation_results")
        self.assertIn("citation_rings_checked", vr)
        self.assertIn("temporal_anomalies_checked", vr)


class TestSummaryStats(unittest.TestCase):

    def test_required_keys(self):
        ss = _get("summary_stats")
        for key in ["total_papers", "total_citations", "unique_authors_raw",
                     "unique_authors_resolved", "orphan_citation_count"]:
            self.assertIn(key, ss)

    def test_headroom_keys(self):
        ss = _get("summary_stats")
        self.assertIn("citation_ring_count", ss)
        self.assertIn("temporal_anomaly_count", ss)


class TestFinalReport(unittest.TestCase):

    def test_anomaly_detection_section(self):
        self.assertIn("anomaly_detection", _get("final_report"))

    def test_citation_rings_in_report(self):
        fr = _get("final_report")
        self.assertIn("citation_rings", fr.get("anomaly_detection", {}))

    def test_all_checks_passed(self):
        fr = _get("final_report")
        self.assertTrue(fr["validation_summary"]["all_checks_passed"])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TEST_CLASSES = [
    TestDataLoading, TestEntityExtraction, TestEntityResolution,
    TestCitationNetwork, TestHeadroomChallenges, TestValidationResults,
    TestSummaryStats, TestFinalReport,
]


def run_all_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run all test classes and return the result."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for tc in ALL_TEST_CLASSES:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    print(f"\nRan {result.testsRun} tests: {len(result.failures)} failures, {len(result.errors)} errors")
    return result
