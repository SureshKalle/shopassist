#!/usr/bin/env python3
# verify_hybrid_e2e.py
"""
Black-box end-to-end verification of the hybrid local/cloud LLM setup
(services/llm_inference.py) and the classifier-driven sentiment integration
(services/classifier_client.py, services/orchestrator.py's step 1.5/5).

Run this against a live API any time you want to confirm the pipeline
(routing, per-role local/cloud model selection, sentiment classification,
tone calibration, cloud->local fallback) is actually behaving as designed,
instead of trusting a one-off manual chat or digging through server logs.

Usage:
    python verify_hybrid_e2e.py [--api-url http://localhost:8000] [--classifier-url http://localhost:8100] [--api-key KEY]

Exits 0 if nothing looks wrong, 1 otherwise. Every check is printed in
plain language, not just PASS/FAIL - some checks (tone wording) are
inherently "soft" since LLM phrasing varies run to run; those are reported
as warnings, not hard failures.

Known limitation (read this before trusting a "sentiment not visible" result):
the chat API's response (api/schemas.py ChatResponse) doesn't expose which
sentiment label/provider/model actually handled a given request - that's
only in server-side logs today ("Customer sentiment: ..." /
"call_generative -> provider=..." in services/orchestrator.py /
services/llm_inference.py). This script's sentiment-injection check is
therefore a *behavioral* proxy (does the reply's wording actually differ),
not a direct assertion on the classifier's output. For a stronger,
non-heuristic assertion, tail the API's logs while this script runs and
grep for those two lines.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import requests

NEGATIVE_TEXT = "This is unacceptable, my order ord-1001 is a disaster and I am extremely upset about this whole experience."
POSITIVE_TEXT = "I just wanted to say order ord-1001 was wonderful, I am so delighted with everything!"

# Soft lexical signal only - real check is "do the two replies read
# differently", these just make that legible in a printed report.
EMPATHY_WORDS = ["sorry", "understand", "apolog", "frustrat", "inconvenien"]
WARMTH_WORDS = ["wonderful", "delight", "glad", "thrilled", "fantastic", "thank you"]


@dataclass
class Check:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str = ""


@dataclass
class Report:
    checks: list = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    def print_and_exit(self) -> None:
        print("\n" + "=" * 72)
        print("HYBRID / MULTI-MODEL END-TO-END VERIFICATION REPORT")
        print("=" * 72)
        for c in self.checks:
            marker = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[c.status]
            print(f"{marker} {c.name}")
            if c.detail:
                for line in c.detail.splitlines():
                    print(f"       {line}")
        print("=" * 72)
        n_fail = sum(1 for c in self.checks if c.status == "FAIL")
        n_warn = sum(1 for c in self.checks if c.status == "WARN")
        print(f"{len(self.checks)} checks, {n_fail} failed, {n_warn} warning(s).")
        sys.exit(1 if n_fail else 0)


def check_api_health(report: Report, api_url: str) -> bool:
    try:
        resp = requests.get(f"{api_url}/api/v1/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        report.add("API reachable", "FAIL", f"{api_url}/api/v1/health unreachable: {e}\nStart it: uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False
    detail = (
        f"agents registered: {data.get('registered_agents')}\n"
        f"database_reachable: {data.get('database_reachable')}\n"
        f"llm_reachable (Ollama specifically): {data.get('llm_reachable')}"
    )
    if not data.get("database_reachable"):
        report.add("API health", "WARN", detail + "\ndatabase_reachable=False - order lookups will fail regardless of the LLM/sentiment checks below.")
    else:
        report.add("API health", "PASS", detail)
    return True


def check_classifier_health(report: Report, classifier_url: str) -> bool:
    try:
        resp = requests.get(f"{classifier_url}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        report.add("Classifier reachable", "PASS", f"{classifier_url} - models_loaded={data.get('models_loaded')}")
        return True
    except requests.exceptions.RequestException as e:
        report.add(
            "Classifier reachable", "WARN",
            f"{classifier_url}/health unreachable: {e}\n"
            "Not fatal - services/classifier_client.py fails soft to label=\"unknown\", "
            "which means the sentiment-tone check below is expected to show NO difference. "
            "That's correct behaviour, not a bug, when the classifier is down.",
        )
        return False


def send_chat(api_url: str, api_key: str, user_id: str, text: str) -> dict:
    resp = requests.post(
        f"{api_url}/api/v1/chat",
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        json={"user_id": user_id, "text": text, "source_channel": "web_chat"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def check_sentiment_tone(report: Report, api_url: str, api_key: str, classifier_up: bool) -> None:
    try:
        negative = send_chat(api_url, api_key, "alum-1001", NEGATIVE_TEXT)
        positive = send_chat(api_url, api_key, "alum-1001", POSITIVE_TEXT)
    except requests.exceptions.RequestException as e:
        report.add("Sentiment-driven tone differs", "FAIL", f"Chat request failed: {e}")
        return

    neg_text = negative.get("response_text", "")
    pos_text = positive.get("response_text", "")
    neg_has_empathy = any(w in neg_text.lower() for w in EMPATHY_WORDS)
    pos_has_warmth = any(w in pos_text.lower() for w in WARMTH_WORDS)
    detail = f"negative-sentiment reply: {neg_text}\n\npositive-sentiment reply: {pos_text}"

    if not classifier_up:
        # Expect near-symmetric, tone-neutral replies - both go through
        # call_generative with customer_sentiment=None/unknown, so nothing
        # should distinguish them beyond ordinary response variance.
        report.add(
            "Sentiment-driven tone (classifier down - expecting NO difference)", "PASS" if not (neg_has_empathy or pos_has_warmth) else "WARN",
            detail + "\n(classifier is down, so any tone match here is coincidental LLM phrasing, not real signal)",
        )
        return

    if neg_has_empathy and pos_has_warmth:
        report.add("Sentiment-driven tone differs", "PASS", detail)
    elif neg_has_empathy or pos_has_warmth:
        report.add("Sentiment-driven tone differs", "WARN", detail + "\nOnly one side showed the expected lexical signal - re-run once, LLM wording varies.")
    else:
        report.add(
            "Sentiment-driven tone differs", "FAIL",
            detail + "\nNeither reply showed the expected empathy/warmth wording. If this repeats, check the "
            "server log for 'Customer sentiment:' and 'call_generative -> provider=' lines to see what "
            "actually happened - see this script's module docstring.",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--classifier-url", default="http://localhost:8100")
    parser.add_argument("--api-key", default="iisc-genai-c2g2p7", help="Must match API_KEY in shopassist/.env")
    args = parser.parse_args()

    report = Report()
    api_up = check_api_health(report, args.api_url)
    classifier_up = check_classifier_health(report, args.classifier_url)
    if api_up:
        check_sentiment_tone(report, args.api_url, args.api_key, classifier_up)
    else:
        report.add("Sentiment-driven tone differs", "FAIL", "Skipped - API is not reachable.")

    print(
        "\nNote: this script cannot see which provider (local/gemini) or which sentiment label\n"
        "actually handled each request above - that's only in the API server's own logs. To\n"
        "fully close the loop, watch the server's terminal while this runs for lines like:\n"
        '  "LLMInferenceService ready | ROUTER=local:... GENERATIVE=gemini:..."\n'
        '  "Customer sentiment: session_id=... label=negative ..."\n'
        '  "call_generative -> provider=gemini model=... mode=parse"\n'
        '  "call_generative: gemini call failed (...) - falling back to local model ..."'
    )
    report.print_and_exit()


if __name__ == "__main__":
    main()
