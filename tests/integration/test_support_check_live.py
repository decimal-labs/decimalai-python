"""Calibrate the narrow Support rubric against known good and bad replies."""
import json
import os
from pathlib import Path

import pytest

from decimalai.agent_checks import DEFAULT_OPENAI_JUDGE, SUPPORT_SKILLS, check_definition, grade_answer

SAMPLES = json.loads((Path(__file__).parents[1] / "fixtures/support_check_calibration.json").read_text())
CASES = {c["id"]: c for c in check_definition(dict(zip(SUPPORT_SKILLS, SUPPORT_SKILLS)))["cases"]}


@pytest.mark.live_llm
@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s["id"])
def test_support_rubric_calibration(sample):
    model = os.environ.get("CHECK_MODEL", DEFAULT_OPENAI_JUDGE)
    if not any(os.environ.get(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")):
        pytest.skip("Requires a provider key for CHECK_MODEL")
    verdict = grade_answer(CASES[sample["case"]], sample["answer"], model)
    # An unavailable judge is a failure of this calibration, never a true negative.
    assert "error" not in verdict, verdict
    assert verdict["passed"] is sample["expected"], verdict
