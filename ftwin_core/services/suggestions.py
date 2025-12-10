from __future__ import annotations

import os
from typing import List, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub


def generate_ai_suggestions(
    *,
    net_monthly: float,
    p10: float,
    p50: float,
    p90: float,
    goal: Optional[float],
    goal_gap: Optional[float],
    horizon_months: int,
    return_mean: float,
    return_vol: float,
    shock_lambda: float,
    shock_mean: float,
    inflation: float,
    savings_rate: Optional[float],
    lifestyle_creep: Optional[float],
    occupation: str,
    breakdown: Optional[dict],
) -> List[str]:
    hf_token = os.environ.get("HF_TOKEN")
    model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    if not hf_token:
        return []

    llm = HuggingFaceHub(repo_id=model, huggingfacehub_api_token=hf_token, model_kwargs={"temperature": 0.4, "max_new_tokens": 200})

    template = """
You are a concise financial coach. Based on the numbers, give 3-5 actionable, specific suggestions. Keep each to one short sentence. No emojis.

Data:
- net_monthly: {net_monthly}
- p10: {p10}
- p50: {p50}
- p90: {p90}
- goal: {goal}
- goal_gap: {goal_gap}
- horizon_months: {horizon_months}
- return_mean: {return_mean}
- return_vol: {return_vol}
- shock_lambda: {shock_lambda}
- shock_mean: {shock_mean}
- inflation: {inflation}
- savings_rate: {savings_rate}
- lifestyle_creep: {lifestyle_creep}
- occupation: {occupation}
- breakdown: {breakdown}

Guidelines:
- Tie suggestions to the gap/percentiles when relevant.
- Include one suggestion about reducing or reallocating spend (use breakdown if present).
- Include one suggestion about increasing income/skill tied to occupation.
- Include one about savings mechanics or buffers (liquidity/risk).
- Be concrete (amounts, duration) when possible. Never restate PII or raw inputs verbatim.
Return suggestions as a plain bullet list with no extra text.
"""
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run(
        net_monthly=net_monthly,
        p10=p10,
        p50=p50,
        p90=p90,
        goal=goal,
        goal_gap=goal_gap,
        horizon_months=horizon_months,
        return_mean=return_mean,
        return_vol=return_vol,
        shock_lambda=shock_lambda,
        shock_mean=shock_mean,
        inflation=inflation,
        savings_rate=savings_rate,
        lifestyle_creep=lifestyle_creep,
        occupation=occupation or "unspecified",
        breakdown=breakdown or {},
    )
    # Simple parse: split lines starting with dash
    ideas = []
    for line in resp.splitlines():
        t = line.strip()
        if t.startswith("-"):
            ideas.append(t.lstrip("-").strip())
    return ideas

