from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# -------------------- Path Config --------------------
INPUT_JSONL = Path(
    "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/textgrad_my/now_demo_icap/optimized_transcripts_with_gradients.jsonl"
)
OUTPUT_JSON = Path(
    "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/textgrad_my/now_demo_icap/evaluation_text_gradients.json"
)
OUTPUT_MD = Path(
    "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/textgrad_my/now_demo_icap/evaluation_text_gradients.md"
)

MODEL = "gpt-5.1"
TIMEOUT = 120
TEMPERATURE = 0.0
MAX_WORKERS = 20


def build_client() -> OpenAI:
    if OpenAI is None:
        raise ImportError("Missing dependency: openai. Install with `pip install openai`.")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is missing. Please set OPENROUTER_API_KEY or OPENAI_API_KEY.")
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
    return OpenAI(api_key=api_key.strip(), base_url=base_url, timeout=TIMEOUT)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def build_eval_items(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        window = str(row.get("window", f"window_{row_idx}"))
        src = str(row.get("input_transcript_path", ""))
        records = row.get("step_records", [])
        if not isinstance(records, list):
            records = []
        valid_records = [rec for rec in records if isinstance(rec, dict)]
        if not valid_records:
            continue
        # 仅评估每个 window 的最后一个 step：
        # 优先按 step 最大值；若 step 缺失则使用列表最后一条。
        try:
            last_rec = max(valid_records, key=lambda r: int(r.get("step", -1)))
            step = int(last_rec.get("step", len(valid_records) - 1))
        except Exception:
            last_rec = valid_records[-1]
            step = int(last_rec.get("step", len(valid_records) - 1))

        text_gradient = str(last_rec.get("text_gradient", "")).strip()
        if not text_gradient:
            continue
        items.append(
            {
                "eval_id": f"{window}__step_{step}",
                "window": window,
                "step": step,
                "input_transcript_path": src,
                "text_gradient": text_gradient,
                "model_name": str(row.get("model_name", "")),
            }
        )
    return items


def build_judge_prompt(item: Dict[str, Any]) -> str:
    return f"""You are a senior educational evaluation expert.

You are evaluating ONE TextGrad feedback artifact: `text_gradient`.
The goal is to judge feedback quality, not to judge transcript content itself.

[Object Being Evaluated]
- text_gradient: critique/instruction text intended to guide transcript rewriting.

[ICAP Reference]
- Standard ICAP hierarchy: Interactive > Constructive > Active > Passive.
- For this task, use ICAP as pedagogical grounding: feedback should not contradict the intended cognitive-engagement direction implied by the transcript critique context.

[Evaluation Dimensions]
A) Accuracy:
   - Judge whether text_gradient correctly identifies concrete problems in the provided context and does not distort what is present.
   - Check whether its pedagogical direction is logically consistent with ICAP-informed improvement (at least not moving in a contradictory direction).
   - Penalize wrong diagnosis, unsupported claims, or ICAP-inconsistent direction.
   - Be conservative: if diagnosis is generic or weakly evidenced, score cannot exceed 3.

B) Alignment & Grounding:
   - Goal Alignment: check whether text_gradient is aligned with the optimization objective:
     clarity/structure, pacing, transitions/signposting, learner engagement, and factual preservation.
   - Grounding Alignment (CRITICAL): check whether claims are grounded in the text_gradient itself (no invented references).
   - Penalize hallucinated references to nonexistent facts/instructions or invented classroom facts.
   - If there is any hallucination/invented reference, alignment_grounding must be <= 2.

C) Actionability:
   - Judge by "Balanced Practicality": reasonable, targeted, executable.
   - Must explicitly include concrete rewrite moves (what to change, how to change, where/when to apply).
   - Penalize BOTH extremes:
     (a) Too vague (slogan-like, no concrete operations)
     (b) Too detailed (over-scripted micromanagement that is unrealistic)
   - Best actionability is middle granularity: clear enough to execute, flexible enough to adapt.
   - If guidance stays slogan-level ("improve clarity", "add transition") without concrete rewrite moves, actionability cannot exceed 3.

[Scoring Constraint]
- Score each dimension 1-5.
- overall_score_5 = round((accuracy + alignment_grounding + actionability) / 3, 2)
- verdict: pass (>=3.75), borderline (3.00-3.74), fail (<3.00)
- Do NOT be generous by default. Start from 3 and move up only with strong evidence.

Return STRICT JSON only:
{{
  "eval_id": "{item["eval_id"]}",
  "scores": {{
    "accuracy": 5,
    "alignment_grounding": 5,
    "actionability": 5
  }},
  "field_quality": {{
    "text_gradient": {{
      "score": 5,
      "comment": "..."
    }}
  }},
  "hallucination_flags": [],
  "evidence_citations": {{
    "accuracy": ["quote/snippet evidence"],
    "alignment_grounding": ["quote/snippet evidence"],
    "actionability": ["quote/snippet evidence"]
  }},
  "major_issues": [],
  "improvement_suggestions": [],
  "dimension_rationales": {{
    "accuracy": "...",
    "alignment_grounding": "...",
    "actionability": "..."
  }},
  "overall_score_5": 5.0,
  "verdict": "pass"
}}

[Current Item]
{json.dumps(item, ensure_ascii=False, indent=2)}
"""


def call_judge(client: OpenAI, item: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_judge_prompt(item)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    content = (response.choices[0].message.content or "{}").strip()
    if content.startswith("```"):
        content = re.sub(r"^```json\s*|```$", "", content, flags=re.MULTILINE).strip()
    return json.loads(content)


def evaluate_one(client: OpenAI, item: Dict[str, Any]) -> Dict[str, Any]:
    try:
        res = call_judge(client, item)
        res = enforce_strict_postcheck(res, item)
        res.update(
            {
                "eval_id": item["eval_id"],
                "window": item["window"],
                "step": item["step"],
                "input_transcript_path": item["input_transcript_path"],
            }
        )
        return res
    except Exception as e:
        return {
            "eval_id": item["eval_id"],
            "window": item["window"],
            "step": item["step"],
            "verdict": "fail",
            "major_issues": [f"Error: {e}"],
            "scores": {"accuracy": 1, "alignment_grounding": 1, "actionability": 1},
            "overall_score_5": 1.0,
        }


def enforce_strict_postcheck(res: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    scores = res.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}

    def clamp(v: Any) -> int:
        try:
            iv = int(v)
        except Exception:
            iv = 1
        return max(1, min(5, iv))

    accuracy = clamp(scores.get("accuracy", 1))
    alignment = clamp(scores.get("alignment_grounding", 1))
    actionability = clamp(scores.get("actionability", 1))

    citations = res.get("evidence_citations", {})
    if not isinstance(citations, dict):
        citations = {}
    acc_ev = citations.get("accuracy", [])
    ali_ev = citations.get("alignment_grounding", [])
    act_ev = citations.get("actionability", [])
    acc_ev = acc_ev if isinstance(acc_ev, list) else []
    ali_ev = ali_ev if isinstance(ali_ev, list) else []
    act_ev = act_ev if isinstance(act_ev, list) else []

    # 保守校准：没有证据引用时上限为3
    if len(acc_ev) == 0:
        accuracy = min(accuracy, 3)
    if len(ali_ev) == 0:
        alignment = min(alignment, 3)
    if len(act_ev) == 0:
        actionability = min(actionability, 3)

    # 有幻觉标记则 Alignment 至多2
    hallucinations = res.get("hallucination_flags", [])
    if isinstance(hallucinations, list) and len(hallucinations) > 0:
        alignment = min(alignment, 2)

    # 过短梯度通常不可执行，压制可执行度
    gradient_text = str(item.get("text_gradient", "")).strip()
    if len(gradient_text) < 120:
        actionability = min(actionability, 3)

    final_scores = {
        "accuracy": accuracy,
        "alignment_grounding": alignment,
        "actionability": actionability,
    }
    overall = round((accuracy + alignment + actionability) / 3, 2)
    verdict = "pass" if overall >= 3.75 else "borderline" if overall >= 3.0 else "fail"

    res["scores"] = final_scores
    res["overall_score_5"] = overall
    res["verdict"] = verdict
    res["evidence_citations"] = {
        "accuracy": acc_ev,
        "alignment_grounding": ali_ev,
        "actionability": act_ev,
    }
    return res


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"count": 0}

    def score(r: Dict[str, Any], key: str) -> float:
        return float(r.get("scores", {}).get(key, 0.0) or 0.0)

    accs = [score(r, "accuracy") for r in results]
    aligns = [score(r, "alignment_grounding") for r in results]
    acts = [score(r, "actionability") for r in results]
    overalls = [float(r.get("overall_score_5", 0) or 0) for r in results]
    verdicts = [str(r.get("verdict", "")).lower() for r in results]
    total = len(results)
    return {
        "count": total,
        "avg_accuracy": round(mean(accs), 3),
        "avg_alignment_grounding": round(mean(aligns), 3),
        "avg_actionability": round(mean(acts), 3),
        "avg_overall_score_5": round(mean(overalls), 3),
        "pass_rate": round(verdicts.count("pass") / total, 3),
        "borderline_rate": round(verdicts.count("borderline") / total, 3),
        "fail_rate": round(verdicts.count("fail") / total, 3),
    }


def to_markdown(report: Dict[str, Any]) -> str:
    agg = report.get("aggregate", {})
    rows = report.get("results", [])
    sum_acc = sum(float(r.get("scores", {}).get("accuracy", 0) or 0) for r in rows)
    sum_align = sum(float(r.get("scores", {}).get("alignment_grounding", 0) or 0) for r in rows)
    sum_act = sum(float(r.get("scores", {}).get("actionability", 0) or 0) for r in rows)
    sum_total = sum(float(r.get("overall_score_5", 0) or 0) for r in rows)
    lines = [
        "# Text Gradient Evaluation Report",
        "",
        "## Aggregate Metrics",
        f"- Count: {agg.get('count')}",
        f"- Avg Accuracy: {agg.get('avg_accuracy')} / 5",
        f"- Avg Alignment & Grounding: {agg.get('avg_alignment_grounding')} / 5",
        f"- Avg Actionability: {agg.get('avg_actionability')} / 5",
        f"- Avg Overall Score: {agg.get('avg_overall_score_5')} / 5",
        f"- Pass/Borderline/Fail Rate: {agg.get('pass_rate')} / {agg.get('borderline_rate')} / {agg.get('fail_rate')}",
        "",
        "## Scoring Table",
        "",
        "| Eval ID | Window | Step | Accuracy | Alignment & Grounding | Actionability | Total(1-5) | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in rows:
        scores = item.get("scores", {})
        lines.append(
            f"| {str(item.get('eval_id','')).replace('|','/')} | {str(item.get('window','')).replace('|','/')} | {item.get('step')} | "
            f"{scores.get('accuracy')} | {scores.get('alignment_grounding')} | {scores.get('actionability')} | "
            f"{item.get('overall_score_5')} | {item.get('verdict')} |"
        )
    lines.append(
        f"| **汇总(平均)** | - | - | **{agg.get('avg_accuracy')}** | **{agg.get('avg_alignment_grounding')}** | "
        f"**{agg.get('avg_actionability')}** | **{agg.get('avg_overall_score_5')}** | - |"
    )
    lines.append(
        f"| **汇总(总和)** | - | - | **{round(sum_acc,3)}** | **{round(sum_align,3)}** | "
        f"**{round(sum_act,3)}** | **{round(sum_total,3)}** | - |"
    )
    return "\n".join(lines)


def main() -> None:
    client = build_client()
    rows = load_jsonl(INPUT_JSONL)
    items = build_eval_items(rows)
    if not items:
        raise ValueError("No valid `text_gradient` items found in JSONL.")

    print(f"Starting parallel text_gradient evaluation. Total items: {len(items)}")
    ordered_results: List[Dict[str, Any]] = [None] * len(items)  # type: ignore

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(evaluate_one, client, item): idx for idx, item in enumerate(items)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            ordered_results[idx] = future.result()

    results = [r for r in ordered_results if r is not None]
    report = {
        "model": MODEL,
        "input_jsonl": str(INPUT_JSONL),
        "evaluation_target": "step_records[*].text_gradient",
        "evaluation_dimensions": ["Accuracy", "Alignment & Grounding", "Actionability"],
        "aggregate": aggregate(results),
        "results": results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(to_markdown(report), encoding="utf-8")
    print(f"Evaluation completed successfully. JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()