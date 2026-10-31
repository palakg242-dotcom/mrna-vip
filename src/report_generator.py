"""Week 4 - Clinical safety report generator (risk model + RAG + LLM -> validated JSON/Markdown).

Pipeline for one patient:
  1. score the patient with the Week 1 risk model
  2. retrieve guideline chunks from the Week 3 index
  3. build a prompt that contains ONLY the risk output and the retrieved chunks (FR-4.2)
  4. ask an LLM (mock / Gemini free tier / Ollama) for a JSON report
  5. validate against the Pydantic schema, retry exactly once on failure (FR-4.3)
  6. enforce the guideline dose limit in code, not in the LLM (FR-4.4)
  7. always attach the disclaimer (FR-4.5) and save .json + .md (FR-4.6)

Run from the repo root:
    python src/report_generator.py                      # 5 sample patients, provider from .env (default: mock)
    python src/report_generator.py --provider gemini    # real LLM (needs GEMINI_API_KEY in .env)
    python src/report_generator.py --list-models        # list Gemini models your key can use
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import DISCLAIMER, MCG_PER_MG  # noqa: E402

OUT_DIR = "outputs"
REPORT_DIR = "outputs/reports"
LLM_LATENCY_LOG = "outputs/llm_latency_log.csv"
PATIENTS_PATH = "data/sample_patients.json"

SYSTEM_PROMPT = f"""You are a clinical safety reporting assistant for a STUDENT ENGINEERING PROJECT. Your output is not real clinical guidance.
You must ONLY use the risk score and the guideline excerpts given in the user message.

Rules you must follow:
1) Never recommend a dose that contradicts the retrieved guideline text.
2) Never invent a clinical metric, statistic, or guideline that is not present in the retrieved guideline excerpts.
3) If the retrieved excerpts do not contain enough information to answer confidently, set Eligible_For_Vaccine to false (conservative default), set Recommended_Dose_mg to 0 and explain why in Reasoning_Steps instead of guessing.
4) Every report must include the Disclaimer field with exactly this text: "{DISCLAIMER}"

Units: the guideline excerpts give doses in micrograms (mcg). Recommended_Dose_mg must be in MILLIGRAMS, where 1 mg = 1000 mcg (so 100 mcg = 0.1 mg).
FDA_Safety_Warnings_Cited must contain at least one sentence copied word-for-word from the excerpts.

Reply with ONLY one JSON object with exactly these keys: Eligible_For_Vaccine (true or false), Recommended_Dose_mg (number), Reasoning_Steps (list of strings), FDA_Safety_Warnings_Cited (list of strings), Disclaimer (string). No markdown and no text outside the JSON."""


class ReportValidationError(Exception):
    """The LLM answer is not a valid, grounded report."""


class LLMCallError(Exception):
    """The LLM provider could not be reached or refused the call."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader (KEY=VALUE per line). Existing environment variables win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


def _normalize(text: str) -> str:
    text = re.sub(r"^\s*-\s+", "", text.strip())
    return re.sub(r"\s+", " ", text).lower().strip(" .")


def mcg_to_mg(mcg: float) -> float:
    return mcg / MCG_PER_MG


# --------------------------------------------------------------------------
# LLM providers
# --------------------------------------------------------------------------
class MockLLM:
    """A fake LLM for testing the pipeline without any key or internet.

    With `responses` it replays those strings (to test malformed answers). Without them it
    applies a few rules copied from the guideline excerpts it is given. It is NOT a real model.
    """

    name = "mock"
    model = "rule-based-mock"

    def __init__(self, responses=None):
        self.responses = list(responses) if responses else None
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        if self.responses:
            return self.responses[min(self.calls, len(self.responses)) - 1]
        return json.dumps(self._rule_based(user))

    @staticmethod
    def _pick(excerpts: str, pattern: str):
        for piece in re.split(r"(?<=[.!?])\s+|\n+", excerpts):
            if re.search(pattern, piece, re.I):
                return re.sub(r"^\s*-\s+", "", piece).strip()
        return None

    def _rule_based(self, user: str) -> dict:
        def num(pattern, default=0.0):
            m = re.search(pattern, user, re.I)
            return float(m.group(1)) if m else default

        age = num(r"(\d+)-year-old")
        conditions = num(r"(\d+) pre-existing")
        dose = num(r"candidate dose ([\d.]+) mcg")
        prior = (re.search(r"prior reaction history: (\w+)", user, re.I) or [None, ""])[1]
        label = (re.search(r"Predicted risk label: (\w+)", user, re.I) or [None, "Low"])[1]
        excerpts = user.split("Relevant guideline excerpts:", 1)[-1]

        eligible, steps, cites = True, [f"The risk model predicts a {label} reaction risk for this patient."], []

        def add_cite(pattern):
            sentence = self._pick(excerpts, pattern)
            if sentence and sentence not in cites:
                cites.append(sentence)

        if dose > 100:
            eligible = False
            steps.append(f"The candidate dose of {dose:g} mcg is above 100 mcg, which needs secondary clinician review.")
            add_cite(r"above 100 micrograms|exceeding 100 micrograms")
        if age >= 65 and conditions >= 2:
            eligible = False
            steps.append("Age 65 or older with two or more pre-existing conditions is flagged for secondary review.")
            add_cite(r"65 and older")
        if conditions >= 3:
            eligible = False
            steps.append("Three or more pre-existing conditions require secondary review.")
            add_cite(r"three or more relevant")
        if prior.lower() == "moderate":
            eligible = False
            steps.append("A prior moderate reaction requires secondary review.")
            add_cite(r"moderate reaction")
        if eligible:
            steps.append("No guideline precaution in the retrieved text blocks this dose.")
            add_cite(r"monitored for at least 15 minutes")
        if not cites:
            add_cite(r"micrograms")
        if not cites:
            cites.append(excerpts.strip().split("\n")[0])
        return {"Eligible_For_Vaccine": eligible,
                "Recommended_Dose_mg": mcg_to_mg(dose) if eligible else 0.0,
                "Reasoning_Steps": steps, "FDA_Safety_Warnings_Cited": cites,
                "Disclaimer": DISCLAIMER}


class GeminiLLM:
    """Google Gemini through the `google-genai` SDK (free tier: key from aistudio.google.com)."""

    name = "gemini"

    def __init__(self, model=None, api_key=None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key or key.startswith("your-"):
            raise LLMCallError("GEMINI_API_KEY not found. Put your key in the .env file (see .env.example).")
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise LLMCallError("Install the SDK first:  uv pip install google-genai") from e
        self._types = types
        self.client = genai.Client(api_key=key)

    def generate(self, system: str, user: str) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system, temperature=0, response_mime_type="application/json")
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(model=self.model, contents=user, config=config)
                return response.text or ""
            except Exception as e:  # SDK raises several error types; classify by message
                message = str(e)
                transient = any(t in message for t in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
                if transient and attempt < 2:
                    time.sleep(20 * (attempt + 1))   # free tier: wait for the per-minute limit to reset
                    continue
                raise LLMCallError(f"Gemini call failed (model '{self.model}'): {message[:300]}") from e
        return ""


class OllamaLLM:
    """A local model served by Ollama (https://ollama.com). Free and offline, but needs a big download."""

    name = "ollama"

    def __init__(self, model=None, url=None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/") + "/api/chat"

    def generate(self, system: str, user: str) -> str:
        payload = {"model": self.model, "stream": False, "format": "json", "options": {"temperature": 0},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)["message"]["content"]
        except (urllib.error.URLError, KeyError, ValueError) as e:
            raise LLMCallError(f"Ollama call failed (is Ollama running and is model '{self.model}' pulled?): {e}") from e


def make_llm(provider=None):
    provider = (provider or os.environ.get("LLM_PROVIDER", "mock")).lower()
    if provider == "mock":
        return MockLLM()
    if provider == "gemini":
        return GeminiLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown provider '{provider}'. Use mock, gemini or ollama.")


def list_gemini_models() -> list:
    """Names of Gemini models your key can call with generateContent."""
    llm = GeminiLLM()
    names = []
    for model in llm.client.models.list():
        actions = getattr(model, "supported_actions", None)
        if actions and "generateContent" not in actions:
            continue
        names.append(str(model.name).replace("models/", ""))
    return names


# --------------------------------------------------------------------------
# Prompt, validation and guideline checks
# --------------------------------------------------------------------------
def build_user_prompt(patient_summary: str, risk_label: str, risk_probability: float, chunks: list) -> str:
    """Only the patient summary, the Module 1 risk output and the Module 3 chunks (FR-4.2)."""
    return (f"Patient summary: {patient_summary}\n"
            f"Predicted risk label: {risk_label} (probability {risk_probability:.2f})\n"
            f"Relevant guideline excerpts:\n" + "\n---\n".join(chunks) +
            "\n\nGenerate the structured clinical safety report now.")


def _strip_to_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def parse_report(text: str, chunks: list):
    """Text -> (validated ClinicalSafetyReport, notes). Raises ReportValidationError if unusable.

    Citations that are not word-for-word in the retrieved chunks are dropped (and noted);
    at least one exact citation must remain.
    """
    from pydantic import ValidationError
    from schemas import ClinicalSafetyReport

    try:
        data = json.loads(_strip_to_json(text))
    except json.JSONDecodeError as e:
        raise ReportValidationError(f"The answer is not valid JSON ({e.msg}).") from e
    try:
        report = ClinicalSafetyReport.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise ReportValidationError(f"The JSON does not match the schema: {problems}") from e

    corpus = _normalize(" ".join(chunks))
    exact = [c for c in report.FDA_Safety_Warnings_Cited if _normalize(c) and _normalize(c) in corpus]
    if not exact:
        raise ReportValidationError("None of FDA_Safety_Warnings_Cited is copied word-for-word "
                                    "from the retrieved guideline excerpts.")
    notes = []
    removed = len(report.FDA_Safety_Warnings_Cited) - len(exact)
    if removed:
        notes.append(f"{removed} citation(s) removed because they were not found word-for-word in the retrieved text.")
    report.FDA_Safety_Warnings_Cited = exact
    return report, notes


def generate_with_retry(llm, system: str, user: str, chunks: list, parser=parse_report):
    """Call the LLM, validate, and on failure retry EXACTLY once (FR-4.3).

    Returns (report, notes, attempts). Raises ReportValidationError after the second failure.
    """
    error = None
    for attempt in (1, 2):
        prompt = user if error is None else (
            f"{user}\n\nYour previous answer was rejected: {error}\n"
            "Return ONLY the corrected JSON object.")
        text = llm.generate(system, prompt)
        try:
            report, notes = parser(text, chunks)
            return report, notes, attempt
        except ReportValidationError as e:
            error = str(e)
    raise ReportValidationError(f"The report failed validation after 1 retry. Last problem: {error}")


_MAX_DOSE = re.compile(r"(?:above|exceeding|over|more than|greater than)\s+(\d+(?:\.\d+)?)\s*(?:micrograms?|mcg|µg)", re.I)


def find_max_dose_mcg(chunks: list):
    """Dose limit (in mcg) stated in the retrieved guideline text, or None if none is stated."""
    values = [float(m.group(1)) for chunk in chunks for m in _MAX_DOSE.finditer(chunk)]
    return min(values) if values else None


def enforce_dose_bound(report, max_mcg, candidate_mcg=None):
    """Reject doses above the guideline limit and make the patient not eligible (FR-4.4).

    Done in code so it does not depend on the LLM following the rule. Returns (report, notes).
    """
    notes = []
    if max_mcg is None:
        return report, notes
    proposed_mcg = report.Recommended_Dose_mg * MCG_PER_MG
    if proposed_mcg > max_mcg + 1e-6:
        report.Recommended_Dose_mg = 0.0
        report.Eligible_For_Vaccine = False
        report.Reasoning_Steps.append(
            f"Safety check: the proposed dose ({proposed_mcg:g} mcg) is above the guideline limit of "
            f"{max_mcg:g} mcg, so it was rejected.")
        notes.append("Proposed dose rejected by the dose-limit check.")
    if candidate_mcg is not None and candidate_mcg > max_mcg + 1e-6 and report.Eligible_For_Vaccine:
        report.Eligible_For_Vaccine = False
        report.Recommended_Dose_mg = 0.0
        report.Reasoning_Steps.append(
            f"Safety check: the candidate dose ({candidate_mcg:g} mcg) is above the guideline limit of "
            f"{max_mcg:g} mcg, so the patient is not marked eligible for this dose.")
        notes.append("Eligibility set to false by the dose-limit check.")
    return report, notes


def conservative_report(reason: str):
    """Fallback when nothing could be retrieved: not eligible, no dose, explained (safety rule 3)."""
    from schemas import ClinicalSafetyReport
    return ClinicalSafetyReport(Eligible_For_Vaccine=False, Recommended_Dose_mg=0.0,
                                Reasoning_Steps=[reason], FDA_Safety_Warnings_Cited=[],
                                Disclaimer=DISCLAIMER)


# --------------------------------------------------------------------------
# Patient scoring, retrieval query, saving
# --------------------------------------------------------------------------
def score_patient(patient: dict, model_path: str = "outputs/risk_model.joblib",
                  pre_path: str = "outputs/preprocessor.joblib"):
    """Week 1 risk model -> (label, probability of High risk)."""
    import joblib
    import pandas as pd

    model, pre = joblib.load(model_path), joblib.load(pre_path)
    row = {c: patient[c] for c in pre["feature_cols"]}
    row["Prior_Reaction_History"] = pre["reaction_order"][patient["Prior_Reaction_History"]]
    X = pd.DataFrame([row])[pre["feature_cols"]]
    X[pre["numeric_cols"]] = pre["scaler"].transform(X[pre["numeric_cols"]])
    probability = float(model.predict_proba(X)[0, 1])
    return ("High" if probability >= 0.5 else "Low"), probability


def patient_summary(p: dict) -> str:
    return (f"{p['Age']:g}-year-old, {p['Pre_Existing_Conditions']} pre-existing condition(s), "
            f"prior reaction history: {p['Prior_Reaction_History']}, "
            f"biomarker level {p['Biomarker_Level']:g}, "
            f"candidate dose {p['Vaccine_Dose_mcg']:g} mcg ({mcg_to_mg(p['Vaccine_Dose_mcg']):g} mg)")


def retrieval_query(p: dict) -> str:
    return (f"Dosing, precautions and monitoring for a {p['Age']:g}-year-old patient with "
            f"{p['Pre_Existing_Conditions']} pre-existing conditions, prior reaction history "
            f"{p['Prior_Reaction_History']}, elevated biomarker, candidate dose "
            f"{p['Vaccine_Dose_mcg']:g} micrograms")


def render_markdown(report, meta: dict) -> str:
    lines = [f"# Clinical Safety Report - {meta['patient_id']}", "",
             f"> {report.Disclaimer}", "",
             f"- **Generated by:** {meta['provider']} ({meta['model']})"
             + (" - a rule-based test double, NOT a real language model" if meta["provider"] == "mock" else ""),
             f"- **Patient:** {meta['summary']}",
             f"- **Risk model (Week 1):** {meta['risk_label']} risk, probability {meta['risk_probability']:.2f}",
             f"- **Eligible for vaccine:** {'Yes' if report.Eligible_For_Vaccine else 'No'}",
             f"- **Recommended dose:** {report.Recommended_Dose_mg:g} mg ({report.Recommended_Dose_mg * MCG_PER_MG:g} mcg)",
             f"- **Latency:** retrieval {meta['rag_latency']:.3f} s, LLM {meta['llm_latency']:.3f} s "
             f"({meta['attempts']} attempt(s))", "", "## Reasoning steps"]
    lines += [f"{i}. {s}" for i, s in enumerate(report.Reasoning_Steps, 1)]
    lines += ["", "## Guideline excerpts cited (word-for-word from the retrieved text)"]
    lines += [f"> {c}" for c in report.FDA_Safety_Warnings_Cited] or ["_None (no guideline text was retrieved)._"]
    if meta["notes"]:
        lines += ["", "## Automatic checks applied"] + [f"- {n}" for n in meta["notes"]]
    lines += ["", "---", report.Disclaimer, ""]
    return "\n".join(lines)


def save_report(report, meta: dict, stem: str, report_dir: str = REPORT_DIR):
    """Write the validated report as .json (schema fields only) and a readable .md (FR-4.6)."""
    os.makedirs(report_dir, exist_ok=True)
    json_path, md_path = os.path.join(report_dir, stem + ".json"), os.path.join(report_dir, stem + ".md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report, meta))
    return json_path, md_path


def _log_llm_latency(path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["timestamp", "patient_id", "provider", "model", "attempts",
                             "rag_latency_seconds", "llm_latency_seconds"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), meta["patient_id"], meta["provider"],
                         meta["model"], meta["attempts"], round(meta["rag_latency"], 4),
                         round(meta["llm_latency"], 4)])


# --------------------------------------------------------------------------
# The full pipeline
# --------------------------------------------------------------------------
def generate_report(patient: dict, llm, retrieve, scorer=score_patient, save: bool = True,
                    report_dir: str = REPORT_DIR, latency_log=LLM_LATENCY_LOG) -> dict:
    """Score -> retrieve -> prompt -> LLM -> validate -> dose check -> save.

    `retrieve(query) -> (chunks, latency)` and `scorer(patient) -> (label, probability)` are
    passed in so tests can replace them.
    """
    label, probability = scorer(patient)
    chunks, rag_latency = retrieve(retrieval_query(patient))
    summary = patient_summary(patient)

    start = time.perf_counter()
    if chunks:
        user = build_user_prompt(summary, label, probability, chunks)
        report, notes, attempts = generate_with_retry(llm, SYSTEM_PROMPT, user, chunks)
    else:
        report = conservative_report("No guideline text could be retrieved, so no decision can be supported "
                                     "by the guidelines. Defaulting to not eligible.")
        notes, attempts = ["No guideline chunks were retrieved; the LLM was not called."], 0
    llm_latency = time.perf_counter() - start

    report, dose_notes = enforce_dose_bound(report, find_max_dose_mcg(chunks), patient["Vaccine_Dose_mcg"])
    report.Disclaimer = DISCLAIMER          # always present, never left to the LLM (FR-4.5)
    meta = {"patient_id": patient["patient_id"], "provider": llm.name, "model": llm.model,
            "summary": summary, "risk_label": label, "risk_probability": probability,
            "rag_latency": rag_latency, "llm_latency": llm_latency, "attempts": attempts,
            "notes": notes + dose_notes, "chunks": chunks}
    if save:
        meta["json_path"], meta["md_path"] = save_report(
            report, meta, f"report_{patient['patient_id']}_{llm.name}", report_dir)
        if latency_log:
            _log_llm_latency(latency_log, meta)
    return {"report": report, "meta": meta}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate validated clinical safety reports.")
    parser.add_argument("--provider", help="mock, gemini or ollama (default: LLM_PROVIDER in .env, else mock)")
    parser.add_argument("--patient", help="only this patient_id from data/sample_patients.json")
    parser.add_argument("--list-models", action="store_true", help="list Gemini models your key can use")
    args = parser.parse_args()

    if args.list_models:
        print("\n".join(list_gemini_models()))
        return

    import rag_index
    with open(PATIENTS_PATH, encoding="utf-8") as f:
        patients = json.load(f)
    if args.patient:
        patients = [p for p in patients if p["patient_id"] == args.patient]
    llm = make_llm(args.provider)
    index = rag_index.load_index()
    retrieve = lambda query: rag_index.query_guidelines(index, query, k=3)  # noqa: E731

    for patient in patients:
        result = generate_report(patient, llm, retrieve)
        report, meta = result["report"], result["meta"]
        print(f"{patient['patient_id']}: eligible={report.Eligible_For_Vaccine}, "
              f"dose={report.Recommended_Dose_mg:g} mg, citations={len(report.FDA_Safety_Warnings_Cited)}, "
              f"attempts={meta['attempts']}, llm={meta['llm_latency']:.2f}s -> {meta['md_path']}")
        if patient is patients[0] and (llm.name != "mock" or not os.path.exists(f"{REPORT_DIR}/sample_report.json")):
            save_report(report, meta, "sample_report")   # the PRD's sample report (a real LLM wins over mock)
    print(f"Provider: {llm.name} ({llm.model}). Latency log: {LLM_LATENCY_LOG}")


if __name__ == "__main__":
    main()
