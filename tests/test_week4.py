"""Tests for Week 4 functional requirements FR-4.1 to FR-4.6 and FR-5.1 to FR-5.3.

Run from the repo root:  python tests/test_week4.py   (or: pytest tests/)

Needs no API key and no internet: the LLM is replaced by the rule-based MockLLM, and the
Week 3 index is replaced by chunks cut from the guideline file. Tests that need `pydantic`
are skipped (with a message) when it is not installed.
"""
import csv
import json
import os
import re
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import dashboard  # noqa: E402
import rag_index  # noqa: E402
import report_generator as rg  # noqa: E402
from constants import DISCLAIMER  # noqa: E402

GUIDELINES = os.path.join(ROOT, "data", "mrna_safety_guidelines.txt")
PATIENTS = os.path.join(ROOT, "data", "sample_patients.json")


def _has_pydantic() -> bool:
    try:
        import pydantic  # noqa: F401
        return True
    except ImportError:
        return False


def _skip(name: str = "") -> None:
    print(f"SKIP  {name or 'a test'} - pydantic is not installed here (it runs on a machine that has it)")


def _chunks() -> list:
    """Stand-in for the Week 3 retrieval result: the real guideline text cut into chunks."""
    tokens = lambda s: sum(1 + len(w) // 7 for w in re.findall(r"\w+|[^\w\s]", s))  # noqa: E731
    return rag_index.chunk_text(open(GUIDELINES, encoding="utf-8").read(), tokens)


def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    return False


GOOD = {"Eligible_For_Vaccine": True, "Recommended_Dose_mg": 0.05,
        "Reasoning_Steps": ["Low risk and the dose is within the guideline range."],
        "FDA_Safety_Warnings_Cited": ["Doses above 100 micrograms are considered high-reactogenicity doses"],
        "Disclaimer": DISCLAIMER}


# ------------------------- Part A: needs no pydantic -------------------------
def test_guideline_dose_limit_is_found_in_the_retrieved_text():
    assert rg.find_max_dose_mcg(_chunks()) == 100.0
    assert rg.find_max_dose_mcg(["Store vials at -25C to -15C."]) is None
    assert rg.mcg_to_mg(100) == 0.1


def test_fr_4_2_prompt_contains_only_risk_output_and_retrieved_chunks():
    prompt = rg.build_user_prompt("58-year-old patient", "High", 0.82, ["CHUNK ONE", "CHUNK TWO"])
    assert "High (probability 0.82)" in prompt and "CHUNK ONE" in prompt and "CHUNK TWO" in prompt
    for rule in ("Never recommend a dose that contradicts", "Never invent a clinical metric",
                 "conservative default", DISCLAIMER):
        assert rule in rg.SYSTEM_PROMPT       # the four required safety constraints are in the system prompt


def test_fr_4_3_exactly_one_retry():
    # bad first answer, good second answer -> 2 calls, success
    llm = rg.MockLLM(responses=["not json", "good"])
    parser = lambda text, chunks: ("REPORT", []) if text == "good" else (_ for _ in ()).throw(  # noqa: E731
        rg.ReportValidationError("bad"))
    report, notes, attempts = rg.generate_with_retry(llm, "sys", "user", [], parser=parser)
    assert (report, attempts, llm.calls) == ("REPORT", 2, 2)

    # always bad -> exactly 2 calls (first try + one retry), then a clear final error
    llm = rg.MockLLM(responses=["bad"])
    assert _raises(rg.ReportValidationError, rg.generate_with_retry, llm, "sys", "user", [],
                   parser=lambda t, c: (_ for _ in ()).throw(rg.ReportValidationError("still bad")))
    assert llm.calls == 2

    # good first answer -> no retry at all
    llm = rg.MockLLM(responses=["good"])
    rg.generate_with_retry(llm, "sys", "user", [], parser=lambda t, c: ("OK", []))
    assert llm.calls == 1


def test_providers_fail_clearly_without_a_key_and_unknown_provider_is_rejected():
    saved = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    try:
        assert _raises(rg.LLMCallError, rg.GeminiLLM)
    finally:
        os.environ.update({k: v for k, v in saved.items() if v})
    assert _raises(ValueError, rg.make_llm, "not-a-provider")
    assert rg.make_llm("mock").name == "mock"


def test_fr_5_dashboard_three_panels_150_dpi_and_data_driven():
    from PIL import Image
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        rag_log, llm_log = os.path.join(d, "rag.csv"), os.path.join(d, "llm.csv")
        with open(rag_log, "w", newline="") as f:
            csv.writer(f).writerows([["timestamp", "question", "k", "latency_seconds", "top_chunk_index"],
                                     ["t", "q", 3, 0.010, 0], ["t", "q", 3, 0.012, 1]])
        with open(llm_log, "w", newline="") as f:
            csv.writer(f).writerows([["timestamp", "patient_id", "provider", "model", "attempts",
                                      "rag_latency_seconds", "llm_latency_seconds"],
                                     ["t", "P1", "mock", "m", 1, 0.01, 0.0002],
                                     ["t", "P1", "gemini", "m", 1, 0.01, 2.5]])
        latencies = dashboard.load_latencies(rag_log, llm_log)
        assert list(latencies.values())[1] == 2.5 and "gemini" in list(latencies)[1]   # real LLM preferred

        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        weights = rng.random((8, 8))
        a, b = os.path.join(d, "a.png"), os.path.join(d, "b.png")
        dashboard.build_dashboard(y, y * 0.6 + rng.random(200) * 0.4, weights, "ACGUACGU", latencies, a)
        dashboard.build_dashboard(y, rng.random(200), weights, "ACGUACGU", latencies, b)   # a worse model
        assert os.path.getsize(a) > 20_000 and open(a, "rb").read() != open(b, "rb").read()   # FR-5.2
        assert Image.open(a).info["dpi"][0] >= 150                                            # FR-5.3
        plt.close("all")

        # FR-5.1: exactly 3 panels, each with a title
        captured = {}
        real_savefig = plt.Figure.savefig
        plt.Figure.savefig = lambda self, *args, **kw: (captured.setdefault("fig", self),
                                                         real_savefig(self, *args, **kw))[1]
        try:
            dashboard.build_dashboard(y, rng.random(200), weights, "ACGUACGU", latencies, a)
        finally:
            plt.Figure.savefig = real_savefig
        axes = [ax for ax in captured["fig"].axes if ax.get_title()]
        assert len(axes) == 3, [ax.get_title() for ax in axes]


# ------------------------- Part B: needs pydantic -------------------------
def test_fr_4_1_schema_has_exactly_the_required_fields():
    if not _has_pydantic():
        return _skip("test_fr_4_1")
    from pydantic import ValidationError
    from schemas import ClinicalSafetyReport

    assert ClinicalSafetyReport(**GOOD).Disclaimer == DISCLAIMER
    assert set(ClinicalSafetyReport.model_fields) == set(GOOD)
    missing = {k: v for k, v in GOOD.items() if k != "Recommended_Dose_mg"}
    assert _raises(ValidationError, ClinicalSafetyReport, **missing)                       # missing field
    assert _raises(ValidationError, ClinicalSafetyReport, **{**GOOD, "Eligible_For_Vaccine": "maybe"})   # wrong type
    assert _raises(ValidationError, ClinicalSafetyReport, **{**GOOD, "Reasoning_Steps": "one string"})
    assert _raises(ValidationError, ClinicalSafetyReport, **{**GOOD, "Extra_Field": 1})    # no extra fields
    assert _raises(ValidationError, ClinicalSafetyReport, **{**GOOD, "Recommended_Dose_mg": -1})


def test_parse_report_checks_json_schema_and_exact_citations():
    if not _has_pydantic():
        return _skip()
    chunks = _chunks()
    report, notes = rg.parse_report(json.dumps(GOOD), chunks)
    assert report.FDA_Safety_Warnings_Cited and not notes

    invented = {**GOOD, "FDA_Safety_Warnings_Cited": [GOOD["FDA_Safety_Warnings_Cited"][0],
                                                      "A made-up guideline about 500 mg doses."]}
    report, notes = rg.parse_report("```json\n" + json.dumps(invented) + "\n```", chunks)
    assert len(report.FDA_Safety_Warnings_Cited) == 1 and notes     # fabricated citation removed and noted

    assert _raises(rg.ReportValidationError, rg.parse_report, "this is prose, not json", chunks)
    assert _raises(rg.ReportValidationError, rg.parse_report, json.dumps({"Eligible_For_Vaccine": True}), chunks)
    only_fake = {**GOOD, "FDA_Safety_Warnings_Cited": ["A made-up guideline."]}
    assert _raises(rg.ReportValidationError, rg.parse_report, json.dumps(only_fake), chunks)


def test_fr_4_4_dose_above_guideline_limit_is_rejected():
    if not _has_pydantic():
        return _skip()
    from schemas import ClinicalSafetyReport

    # the LLM wrongly proposes 0.12 mg (= 120 mcg): rejected in code
    report, notes = rg.enforce_dose_bound(ClinicalSafetyReport(**{**GOOD, "Recommended_Dose_mg": 0.12}), 100.0, 120)
    assert report.Eligible_For_Vaccine is False and report.Recommended_Dose_mg == 0.0 and notes
    # the LLM says "eligible" for a 120 mcg candidate: still not eligible
    report, _ = rg.enforce_dose_bound(ClinicalSafetyReport(**GOOD), 100.0, 120)
    assert report.Eligible_For_Vaccine is False
    # within the limit: unchanged
    report, notes = rg.enforce_dose_bound(ClinicalSafetyReport(**GOOD), 100.0, 50)
    assert report.Eligible_For_Vaccine is True and report.Recommended_Dose_mg == 0.05 and not notes


def test_fr_4_5_and_4_6_five_sample_reports_with_disclaimer_and_file_pairs():
    if not _has_pydantic():
        return _skip()
    chunks = _chunks()
    patients = json.load(open(PATIENTS, encoding="utf-8"))
    assert len(patients) == 5
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        results = {}
        for p in patients:
            results[p["patient_id"]] = rg.generate_report(
                p, rg.MockLLM(), retrieve=lambda q: (chunks, 0.01), scorer=lambda patient: ("High", 0.8),
                report_dir=d, latency_log=os.path.join(d, "llm_latency.csv"))
        for pid, res in results.items():
            report = res["report"]
            assert report.Disclaimer == DISCLAIMER                       # FR-4.5
            assert report.FDA_Safety_Warnings_Cited                      # DoD: cites a retrieved excerpt
            assert os.path.exists(res["meta"]["json_path"]) and os.path.exists(res["meta"]["md_path"])   # FR-4.6
            saved = json.load(open(res["meta"]["json_path"], encoding="utf-8"))
            assert saved["Disclaimer"] == DISCLAIMER and set(saved) == set(GOOD)
            assert DISCLAIMER in open(res["meta"]["md_path"], encoding="utf-8").read()
        assert results["P001"]["report"].Eligible_For_Vaccine is False   # 120 mcg candidate (FR-4.4)
        assert results["P002"]["report"].Eligible_For_Vaccine is True    # 34 y, 0 conditions, 50 mcg
        assert results["P003"]["report"].Eligible_For_Vaccine is False   # 65+ with two conditions
        assert results["P004"]["report"].Eligible_For_Vaccine is False   # three conditions
        assert len(open(os.path.join(d, "llm_latency.csv")).readlines()) == 1 + 5


def test_no_retrieved_text_gives_a_conservative_report_without_calling_the_llm():
    if not _has_pydantic():
        return _skip()
    llm = rg.MockLLM()
    p = json.load(open(PATIENTS, encoding="utf-8"))[1]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        res = rg.generate_report(p, llm, retrieve=lambda q: ([], 0.0), scorer=lambda x: ("Low", 0.1),
                                 report_dir=d, latency_log=None)
    assert llm.calls == 0 and res["report"].Eligible_For_Vaccine is False
    assert res["report"].Recommended_Dose_mg == 0.0 and res["report"].Disclaimer == DISCLAIMER


def test_a_malformed_first_answer_is_retried_once_through_the_real_parser():
    if not _has_pydantic():
        return _skip()
    llm = rg.MockLLM(responses=["I think the patient is fine.", json.dumps(GOOD)])
    report, notes, attempts = rg.generate_with_retry(llm, rg.SYSTEM_PROMPT, "user", _chunks())
    assert attempts == 2 and llm.calls == 2 and report.Eligible_For_Vaccine


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} Week 4 tests passed.")
