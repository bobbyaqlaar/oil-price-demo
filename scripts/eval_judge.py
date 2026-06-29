"""
eval_judge.py — Shared LLM-judge invocation logic.

Factored out of run-evals.py so shadow-eval.py (P1c, samples live Phoenix
production spans instead of the golden dataset) can reuse the exact same
judge-prompting/JSON-parsing path rather than duplicating it. Both callers
hit the same `cost_router.call()` LLM dispatch.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def judge_prompt(
    instructions: str,
    historical_text: str,
    input_text: str,
    expected_tool: str,
    reference_output: str,
    actual_output: str,
) -> str:
    return f"""{instructions}

{historical_text}

=== CASE TO EVALUATE ===
INPUT: {input_text}
EXPECTED TOOL: {expected_tool}
REFERENCE OUTPUT: {reference_output}
ACTUAL OUTPUT:
{actual_output}

Respond with ONLY a JSON object:
{{
  "correctness": 0 or 1,
  "tool_accuracy": 0 or 1,
  "quality_notes": "<brief observation>",
  "score": 0.0..1.0
}}"""


def run_judge(prompt: str, judge_model: str) -> dict[str, Any]:
    """
    Invoke the configured judge model on a prompt built by judge_prompt()
    (or an equivalent), parse its JSON verdict.

    Returns a dict with correctness/tool_accuracy/score/quality_notes, and
    an "error" key set if the judge call or parse failed.
    """
    from cost_router import call as llm_call

    try:
        raw = llm_call(
            prompt,
            system="You are a strict technical evaluator. Respond with JSON only.",
            task_type="review",
            force_model=judge_model,
        )
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        scored = json.loads(m.group(0)) if m else {"correctness": 0, "tool_accuracy": 0, "score": 0.0}
    except Exception as exc:
        scored = {"correctness": 0, "tool_accuracy": 0, "score": 0.0, "error": str(exc)}
    return scored


def judge_case(
    case: dict,
    criteria: dict,
    judge_model: str,
    project_response: Optional[str] = None,
) -> dict[str, Any]:
    """
    Score one golden case against the configured judge. Used by run-evals.py.
    """
    historical = criteria.get("historical_learnings", [])
    historical_text = "\n".join(f"- {l}" for l in historical) if historical else "(none yet)"

    prompt = judge_prompt(
        instructions=criteria.get("instructions", ""),
        historical_text=historical_text,
        input_text=case["input"],
        expected_tool=case.get("expected_tool", "any"),
        reference_output=case.get("reference_output", "(none)"),
        actual_output=project_response or "",
    )
    return run_judge(prompt, judge_model)
