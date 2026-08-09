"""Claim-coverage LLM judge for MCP-Atlas rewards.

Reimplements the official scorer (``services/scoring/score_claims.py`` in the MCP-Atlas repo)
as a per-trajectory async call: one judge request per ground-truth claim, scored
fulfilled=1.0 / partially_fulfilled=0.5 / not_fulfilled=0.0, averaged into a coverage score
in [0, 1]. The judge prompt and score mapping are copied verbatim so rewards match the
benchmark's leaderboard metric.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI

# Verbatim from MCP-Atlas services/scoring/score_claims.py::_get_single_claim_evaluation_prompt.
CLAIM_EVALUATION_PROMPT = """You are evaluating how well a model's response addresses a specific expert-defined claim.
SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed. The response covers all key details.
- partially_fulfilled: Claim is partially addressed. The response covers some but not all key details.
- not_fulfilled: Claim is not addressed. The response does not include any key details.
NUMERICAL COMPARISON GUIDELINES:
- For numerical values, use reasonable approximation thresholds:
  * Exact match NOT required for decimals
  * Values within 5% of the claimed number are considered matching
  * For percentages, ±1 percentage points is acceptable
  * Round to appropriate significant figures based on context
- Consider the precision appropriate to the domain:
  * Scientific measurements may need higher precision
  * General statistics/estimates can have looser matching
  * Financial figures should match to reasonable business precision (e.g., millions/billions don't need exact cents)
- If a number is expressed differently but mathematically equivalent (e.g., "0.5" vs "50%" vs "half"), consider it a match
CLAIM TO EVALUATE:
{claim}
MODEL RESPONSE TO ANALYZE:
{response}
INSTRUCTIONS:
1. Determine if the core requirement of the claim is met in the response
2. Check if all key components from the claim appear substantively in the response
   - For numerical values, apply the flexible matching guidelines above
   - Focus on whether the same magnitude and meaning are conveyed
3. Assign the appropriate coverage_outcome
4. Provide specific justification referencing what was/wasn't covered
   - When numbers differ slightly, note if they're within acceptable range
5. Provide a confidence level (0.0-1.0) for your assessment
Be rigorous but fair in your assessment. Focus on whether the response conveys the same information as the claim, not on exact numerical precision unless precision is critical to the claim's meaning."""

COVERAGE_TO_SCORE = {"fulfilled": 1.0, "partially_fulfilled": 0.5, "not_fulfilled": 0.0}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_text": {"type": "string"},
        "coverage_outcome": {"type": "string", "enum": ["fulfilled", "partially_fulfilled", "not_fulfilled"]},
        "justification": {"type": "string"},
        "confidence_level": {"type": "number"},
    },
    "required": ["claim_text", "coverage_outcome", "justification", "confidence_level"],
}

# Responses longer than this are truncated before judging (matches the official scorer).
MAX_RESPONSE_CHARS = 500_000


class JudgeError(Exception):
    """Raised when the judge endpoint fails after retries; callers should mask the trajectory."""


class ClaimCoverageJudge:
    def __init__(self, judge_cfg: Dict[str, Any]):
        """
        Args:
            judge_cfg: dict with keys ``model``, ``base_url``, ``api_key``, ``timeout_seconds``,
                ``max_retries``, ``max_concurrency``. ``base_url``/``api_key``/``model`` fall back
                to the EVAL_LLM_BASE_URL / EVAL_LLM_API_KEY / EVAL_LLM_MODEL environment variables.
        """
        base_url = judge_cfg.get("base_url") or os.environ.get("EVAL_LLM_BASE_URL")
        api_key = judge_cfg.get("api_key") or os.environ.get("EVAL_LLM_API_KEY")
        model = judge_cfg.get("model") or os.environ.get("EVAL_LLM_MODEL")
        if not base_url or not model:
            raise ValueError(
                "Judge endpoint not configured. Set mcp_atlas_config.judge.base_url/model (or "
                "EVAL_LLM_BASE_URL/EVAL_LLM_MODEL env vars) to a real LLM endpoint. The judge must "
                "not point at the policy being trained."
            )
        self.model = model
        self.max_retries = int(judge_cfg.get("max_retries", 2))
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "dummy",
            timeout=float(judge_cfg.get("timeout_seconds", 60)),
            max_retries=0,
        )
        self._semaphore = asyncio.Semaphore(int(judge_cfg.get("max_concurrency", 16)))

    async def _evaluate_single_claim(self, claim: str, response: str) -> float:
        prompt = CLAIM_EVALUATION_PROMPT.format(claim=claim, response=response[:MAX_RESPONSE_CHARS])
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._semaphore:
                    completion = await self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "claim_evaluation", "schema": RESPONSE_SCHEMA},
                        },
                    )
                return self._parse_outcome(completion.choices[0].message.content or "")
            except Exception as e:  # noqa: BLE001 - any endpoint/parse error is retried, then raised
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
        raise JudgeError(f"Judge failed after {self.max_retries + 1} attempts: {last_error}")

    @staticmethod
    def _parse_outcome(content: str) -> float:
        """Parse the judge reply into a score; tolerates markdown fences and loose JSON."""
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
        try:
            outcome = json.loads(text).get("coverage_outcome", "")
        except (json.JSONDecodeError, AttributeError):
            match = re.search(r"(partially_fulfilled|not_fulfilled|fulfilled)", text)
            if match is None:
                raise JudgeError(f"Unparseable judge reply: {content[:200]!r}")
            outcome = match.group(1)
        if outcome not in COVERAGE_TO_SCORE:
            raise JudgeError(f"Unknown coverage_outcome: {outcome!r}")
        return COVERAGE_TO_SCORE[outcome]

    async def score(self, claims: List[str], response: str) -> float:
        """Return the coverage score in [0, 1] for a final response against ground-truth claims.

        Empty responses score 0.0 without calling the judge (matches the official scorer).
        Raises JudgeError if any claim evaluation fails after retries.
        """
        if not claims:
            raise ValueError("claims must be non-empty")
        if not response or not response.strip() or response.startswith("ERROR:"):
            return 0.0
        scores = await asyncio.gather(*(self._evaluate_single_claim(c, response) for c in claims))
        score = round(sum(scores) / len(scores), 3)
        logger.debug(f"Judge coverage: {score} over {len(claims)} claims")
        return score
