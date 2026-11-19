"""Seed an isolated, key-free customer-support evaluation and serve the demo API."""

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass

import httpx
import uvicorn

BASELINE = "Match one primary policy keyword."
CHALLENGER = "Match primary policy keywords and common customer aliases."
ESCALATE = "Contact support for help with this request."

POLICIES = {
    "refund": (
        ("return", "money back", "reimbursement"),
        "Request a refund within 30 days of delivery. Refunds go to the original payment method.",
    ),
    "password": (
        ("locked out", "sign in", "reset link"),
        "Select Forgot password on the sign-in page. Use the reset link we email you.",
    ),
    "invoice": (
        ("receipt", "billing statement", "tax document"),
        "Open Settings, then Billing and Invoices. Select the invoice and download the PDF.",
    ),
    "email": (
        ("billing address", "where bills go", "accounts mailbox"),
        "Change your billing email under Settings, then Billing. It applies to future invoices.",
    ),
    "shipping": (
        ("parcel", "delivery tracking", "package"),
        "Open Orders and select Track package to find the carrier and tracking number.",
    ),
    "cancel": (
        ("stop renewal", "unsubscribe", "end subscription"),
        "Open Settings, then Subscription, and select Cancel renewal. Access lasts until the billing period ends.",
    ),
}


@dataclass(frozen=True)
class Case:
    question: str
    expected_terms: tuple[str, ...]


# Fixed independently of the evaluation loop. Terms are checked after generation.
CASES = (
    Case("How do I request a refund?", ("30 days", "original payment method")),
    Case("What is the deadline for a return?", ("30 days", "delivery")),
    Case("Can I get my money back?", ("refund", "original payment method")),
    Case("How do I reset my password?", ("Forgot password", "reset link")),
    Case("I'm locked out and can't get into my account.", ("Forgot password", "email")),
    Case("Where do I find the sign in help?", ("Forgot password", "sign-in")),
    Case("How do I download an invoice?", ("Billing", "PDF")),
    Case("Where can I see my receipt?", ("Invoices", "PDF")),
    Case("Where is my billing statement?", ("Billing", "download")),
    Case("How do I change my billing email?", ("Settings", "future invoices")),
    Case("Can I change where bills go?", ("billing email", "future invoices")),
    Case("I need to set a new accounts mailbox.", ("Settings", "future invoices")),
    Case("How can I get shipping details?", ("Track package", "carrier")),
    Case("Did my parcel leave yet?", ("Orders", "tracking number")),
    Case("Where is my package?", ("Track package", "tracking number")),
    Case("How do I cancel at the end of the month?", ("Cancel renewal", "billing period ends")),
    Case("I want to unsubscribe.", ("Subscription", "Cancel renewal")),
    Case("Can I get a refund on day 31 after delivery?", ("refund", "30 days")),
)
EXPECTED_TERMS = {case.question: case.expected_terms for case in CASES}


def answer(question: str, *, use_aliases: bool) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold())
    for keyword, (aliases, response) in POLICIES.items():
        terms = (keyword, *aliases) if use_aliases else (keyword,)
        if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms):
            return response
    return ESCALATE


def fact_hits(question: str, response: str) -> int:
    terms = EXPECTED_TERMS.get(question, ("contact support",))
    return sum(term.casefold() in response.casefold() for term in terms)


class DemoClient:
    """Deterministic generation and scoring behind the normal LLM interface."""

    async def complete(self, prompt: str, response_format: dict | None = None, **kwargs):
        from neuralops.services.llm_client import LLMBackend, LLMResponse, _count_tokens

        if response_format:
            question, _, evaluation = prompt.removeprefix("User input:\n").partition(
                "\n\nResponse to evaluate:\n"
            )
            response = evaluation.partition("\n\nCriteria to focus on:")[0]
            expected = EXPECTED_TERMS.get(question, ("contact support",))
            coverage = fact_hits(question, response) / len(expected)
            content = json.dumps({
                "relevance": 2 + 8 * coverage,
                "accuracy": 1 + 9 * coverage,
                "clarity": 8,
                "safety": 9,
                "confidence": 1.0,
                "reasoning": f"Offline rubric matched {coverage:.0%} of expected terms.",
            })
            model = "offline-term-rubric"
        else:
            instruction, separator, question = prompt.partition("\n\n")
            content = answer(question if separator else prompt, use_aliases=instruction == CHALLENGER)
            model = "policy-keywords-and-aliases" if instruction == CHALLENGER else "policy-keywords"

        prompt_tokens = _count_tokens((kwargs.get("system") or "") + prompt)
        completion_tokens = _count_tokens(content)
        return LLMResponse(
            content=content,
            model=model,
            backend=LLMBackend.FIXTURE,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=0.0,
            latency_ms=0.0,
        )


async def seed(app) -> dict:
    """Load the fixed dataset through the same API the dashboard uses."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://demo") as api:
        async def post(path: str, data: dict) -> dict:
            response = await api.post(path, json=data)
            response.raise_for_status()
            return response.json()

        prompt = await post("/api/v1/prompts/", {
            "name": "Customer support",
            "description": "Compare keyword matching with customer-language aliases for six support policies.",
        })
        prompt_path = f"/api/v1/prompts/{prompt['id']}"
        for version, instruction in (("1.0.0", BASELINE), ("2.0.0", CHALLENGER)):
            await post(f"{prompt_path}/versions/", {
                "version": version,
                "content": instruction + "\n\n{input}",
            })
        await post(f"{prompt_path}/activate/1.0.0", {})

        experiment = await post("/api/v1/experiments/", {
            "name": "Support policy answers · offline benchmark",
            "prompt_id": prompt["id"],
            "version_a": "1.0.0",
            "version_b": "2.0.0",
            "planned_trials": len(CASES),
        })
        experiment_path = f"/api/v1/experiments/{experiment['id']}"
        hits = {"a": 0, "b": 0}
        for case in CASES:
            trial = await post(f"{experiment_path}/trials/", {"user_input": case.question})
            for arm in hits:
                hits[arm] += fact_hits(case.question, trial[f"output_{arm}"]) == len(case.expected_terms)

        response = await api.get(f"{experiment_path}/significance")
        response.raise_for_status()
        return {
            "cases": len(CASES),
            "complete_fact_matches": hits,
            **{key: value for key, value in response.json().items() if key != "experiment_id"},
        }


async def run(*, benchmark: bool, port: int) -> None:
    from neuralops.api.dependencies import get_llm_client
    from neuralops.api.main import app
    from neuralops.core.database import Base, engine
    from neuralops.core.logging import configure_logging

    configure_logging(level="WARNING", json_logs=False)
    client = DemoClient()
    app.dependency_overrides[get_llm_client] = lambda: client
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        report = await seed(app)
        if benchmark:
            print(json.dumps(report, indent=2))
        else:
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port))
            await server.serve()
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="Print the benchmark and exit")
    parser.add_argument("--port", type=int, default=8000, help="Loopback API port")
    args = parser.parse_args()
    os.environ.update({
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "NEURALOPS_ENVIRONMENT": "demo",
        "NEURALOPS_PORT": str(args.port),
        "LLM_PROVIDER": "fixture",
        "JUDGE_BACKEND": "fixture",
        "JUDGE_MODEL": "offline-term-rubric",
        "EXPERIMENT_SIGNIFICANCE_LEVEL": "0.05",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
    })
    asyncio.run(run(benchmark=args.benchmark, port=args.port))


if __name__ == "__main__":
    main()
