"""
Evaluate reflection signals (last step per knowledge point) with one unified ICAP prompt.
No dean/calibration stage: directly score three dimensions aligned with eval2.py.
"""

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from openai import OpenAI

_openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
_openai_key = os.getenv("OPENAI_API_KEY", "").strip()

if _openrouter_key:
    MODEL = "openai/gpt-4o-mini"
    client = OpenAI(api_key=_openrouter_key, base_url="https://openrouter.ai/api/v1")
elif _openai_key:
    MODEL = "gpt-4o-mini"
    client = OpenAI(api_key=_openai_key)
else:
    raise ValueError(
        "API key is not set. Please set OPENROUTER_API_KEY or OPENAI_API_KEY."
    )

TIMEOUT = 60
TEMPERATURE = 0
MAX_WORKERS = max(1, int(os.getenv("REPORT_EVAL_MAX_WORKERS", "8")))

INPUT_JSONL_PATH = r""
OUTPUT_JSON_PATH = r""
OUTPUT_MD_PATH = r""

TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0
PRICE_INPUT_PER_1K = 0.00015
PRICE_OUTPUT_PER_1K = 0.00060
COUNTER_LOCK = threading.Lock()


def safe_json_loads(text: str) -> Dict:
    if not text or not text.strip():
        raise ValueError("Empty LLM output")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")
    return json.loads(match.group(0))


def build_judge_prompt(
    knowledge_point_id: object,
    topic: str,
    reflection_signal: str,
    transcript_text: str,
) -> str:
    return f"""You are a senior educational evaluation expert. Please objectively evaluate the following reflection signal for one knowledge point.

[ICAP Reference]
- Interactive (I): two or more students co-construct by responding to each other.
- Constructive (C): a student individually generates explanations/new links beyond simple repeat.
- Active (A): visible effort without clear idea generation.
- Passive (P): mainly receiving information.
- Cognitive depth ordering: I > C > A > P.

[Three Major Evaluation Dimensions and Scoring Standards (1-5 points)]
A) Accuracy:
   - Judge whether reflection_signal correctly identifies concrete problems in the provided context and does not distort what is present.
   - Check whether its pedagogical direction is logically consistent with ICAP-informed improvement (at least not moving in a contradictory direction).
   - Penalize wrong diagnosis, unsupported claims, or ICAP-inconsistent direction.
   - Be conservative: if diagnosis is generic or weakly evidenced, accuracy cannot exceed 3.
B) Alignment & Grounding:
   - Goal Alignment: check whether reflection_signal is aligned with optimization goals:
     clarity/structure, pacing, transitions/signposting, learner engagement, and factual preservation.
   - Grounding Alignment (CRITICAL): check whether claims are grounded in reflection_signal and transcript context (no invented references).
   - Penalize hallucinated references to nonexistent facts/instructions or invented classroom facts.
   - If there is any hallucination/invented reference, alignment_grounding must be <= 2.
C) Actionability:
   - Judge by "Balanced Practicality": reasonable, targeted, and executable.
   - Must explicitly include concrete rewrite moves (what to change, how to change, where/when to apply).
   - Penalize BOTH extremes:
     (a) Too vague (slogan-like, no concrete operations)
     (b) Too detailed (over-scripted micromanagement that is unrealistic)
   - Best actionability is middle granularity: clear enough to execute, flexible enough to adapt.
   - If guidance stays slogan-level without concrete rewrite moves, actionability cannot exceed 3.

[Actionability scoring anchor]
- 5: Strongly targeted + clear actor/action/timing + practical fit.
- 4: Good and executable, minor gaps.
- 3: Partly executable, somewhat generic or somewhat over-engineered.
- 2: Mostly vague or impractical.
- 1: Not actionable.

[Scoring Constraints]
- overall_score_5 = round((accuracy + alignment_grounding + actionability) / 3, 2)
- verdict: pass (>=3.75), borderline (3.00-3.74), fail (<3.00)

Output strictly JSON only:
{{
  "knowledge_point_id": {json.dumps(knowledge_point_id, ensure_ascii=False)},
  "topic": {json.dumps(topic, ensure_ascii=False)},
  "scores": {{ "accuracy": 5, "alignment_grounding": 5, "actionability": 5 }},
  "dimension_rationales": {{ "accuracy": "...", "alignment_grounding": "...", "actionability": "..." }},
  "major_issues": [],
  "improvement_suggestions": [],
  "overall_score_5": 5.0,
  "verdict": "pass"
}}

[Reflection Signal]
{reflection_signal}

[Transcript Context]
{transcript_text if transcript_text.strip() else "[No transcript provided]"}
"""


def call_llm(prompt: str) -> Dict:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, REQUEST_COUNT
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
    )
    usage = getattr(response, "usage", None)
    with COUNTER_LOCK:
        if usage is not None:
            TOTAL_PROMPT_TOKENS += int(getattr(usage, "prompt_tokens", 0) or 0)
            TOTAL_COMPLETION_TOKENS += int(getattr(usage, "completion_tokens", 0) or 0)
        REQUEST_COUNT += 1
    content = response.choices[0].message.content or "{}"
    return safe_json_loads(content)


def evaluate_feedback(
    knowledge_point_id: object,
    topic: str,
    reflection_signal: str,
    transcript_text: str,
) -> Dict:
    prompt = build_judge_prompt(
        knowledge_point_id=knowledge_point_id,
        topic=topic,
        reflection_signal=reflection_signal,
        transcript_text=transcript_text[:8000],
    )
    result = call_llm(prompt)
    result["knowledge_point_id"] = knowledge_point_id
    result["topic"] = topic
    return result


def evaluate_group(group_label: str, rows: List[Dict]) -> Dict:
    if not rows:
        return {"group_label": group_label, "result": {"error": "empty grouped records"}}

    target = max(
        [r for r in rows if isinstance(r, dict)],
        key=lambda x: int(x.get("step", -1)),
        default={},
    )

    reflection_signal = str(target.get("reflection_signal", "")).strip()
    if not reflection_signal:
        return {
            "group_label": group_label,
            "result": {"error": "empty last-round reflection_signal"},
        }

    transcript_path = str(target.get("transcript_path", "")).strip()
    transcript_text = ""
    if transcript_path and os.path.isfile(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as tf:
                transcript_text = tf.read().strip()
        except Exception:
            transcript_text = ""

    try:
        evaluation = evaluate_feedback(
            knowledge_point_id=target.get("knowledge_point_id"),
            topic=str(target.get("topic", "")),
            reflection_signal=reflection_signal,
            transcript_text=transcript_text,
        )
    except Exception as e:
        return {"group_label": group_label, "result": {"error": str(e)}}

    return {
        "group_label": group_label,
        "result": {
            "knowledge_point_id": target.get("knowledge_point_id"),
            "topic": target.get("topic", ""),
            "last_step": target.get("step"),
            "reflection_signal": reflection_signal,
            **evaluation,
        },
    }


def to_markdown_table(all_results: Dict[str, Dict]) -> str:
    valid_rows: List[Dict] = []
    for label, result in all_results.items():
        if not isinstance(result, dict) or "error" in result:
            continue
        scores = result.get("scores", {})
        try:
            accuracy = float(scores.get("accuracy", 0) or 0)
            alignment = float(scores.get("alignment_grounding", 0) or 0)
            actionability = float(scores.get("actionability", 0) or 0)
            overall = float(result.get("overall_score_5", 0) or 0)
        except (TypeError, ValueError):
            continue
        valid_rows.append(
            {
                "label": label,
                "knowledge_point_id": result.get("knowledge_point_id", ""),
                "topic": str(result.get("topic", "")).replace("|", "/"),
                "accuracy": accuracy,
                "alignment_grounding": alignment,
                "actionability": actionability,
                "overall_score_5": overall,
                "verdict": result.get("verdict", ""),
            }
        )

    valid_rows.sort(key=lambda x: (str(x.get("label", "")), str(x.get("knowledge_point_id", ""))))

    if not valid_rows:
        return "# Reflection Evaluation Table\n\nNo valid rows."

    avg_accuracy = sum(r["accuracy"] for r in valid_rows) / len(valid_rows)
    avg_alignment = sum(r["alignment_grounding"] for r in valid_rows) / len(valid_rows)
    avg_actionability = sum(r["actionability"] for r in valid_rows) / len(valid_rows)
    avg_overall = sum(r["overall_score_5"] for r in valid_rows) / len(valid_rows)

    lines = [
        "# Reflection Evaluation Table",
        "",
        "| Group | KP | Topic | Accuracy | Alignment & Grounding | Actionability | Overall(1-5) | Verdict |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in valid_rows:
        lines.append(
            f"| {row['label']} | {row['knowledge_point_id']} | {row['topic']} | "
            f"{row['accuracy']:.2f} | {row['alignment_grounding']:.2f} | {row['actionability']:.2f} | "
            f"{row['overall_score_5']:.2f} | {row['verdict']} |"
        )
    lines.append(
        f"| **平均值** | - | - | **{avg_accuracy:.2f}** | **{avg_alignment:.2f}** | "
        f"**{avg_actionability:.2f}** | **{avg_overall:.2f}** | - |"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    if not os.path.isfile(INPUT_JSONL_PATH):
        raise FileNotFoundError(f"[ERROR] INPUT_JSONL_PATH not found: {INPUT_JSONL_PATH}")

    by_group: Dict[str, List[Dict]] = {}
    with open(INPUT_JSONL_PATH, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[SKIP] line {line_idx}: invalid json -> {e}")
                continue
            kp_id = rec.get("knowledge_point_id")
            key = f"kp_{kp_id}" if isinstance(kp_id, int) else str(rec.get("window", f"unknown_window_{line_idx}"))
            by_group.setdefault(key, []).append(rec)

    def _group_key(label: str):
        m = re.match(r"kp_(\d+)$", label)
        if m:
            return (0, int(m.group(1)))
        m2 = re.match(r"(\d+)m_(\d+)m", label)
        if m2:
            return (1, int(m2.group(1)) * 1000 + int(m2.group(2)))
        return (10**9, 10**9)

    labels = sorted(by_group.keys(), key=_group_key)
    all_results: Dict[str, Dict] = {}
    print(f"[START] grouped_items={len(labels)} workers={MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_label = {
            executor.submit(evaluate_group, label, by_group.get(label, [])): label
            for label in labels
        }
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            print(f"\n[EVAL] {label}")
            try:
                payload = future.result()
                result = payload.get("result", {})
                all_results[label] = result
                if "error" in result:
                    print(f"  [SKIP] {result['error']}")
                else:
                    print(
                        f"  [DONE] last_step={result.get('last_step')}, "
                        f"overall_score_5={result.get('overall_score_5', '?')}, "
                        f"verdict={result.get('verdict', '?')}"
                    )
            except Exception as e:
                print(f"  [FAIL] {e}")
                all_results[label] = {"error": str(e)}

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    table_md = to_markdown_table(all_results)
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(table_md)

    if REQUEST_COUNT > 0:
        total_prompt_k = TOTAL_PROMPT_TOKENS / 1000.0
        total_completion_k = TOTAL_COMPLETION_TOKENS / 1000.0
        total_tokens = TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS
        cost_input = total_prompt_k * PRICE_INPUT_PER_1K
        cost_output = total_completion_k * PRICE_OUTPUT_PER_1K
        total_cost = cost_input + cost_output
        print(
            f"\n[TOKEN] total_prompt_tokens={TOTAL_PROMPT_TOKENS}, "
            f"total_completion_tokens={TOTAL_COMPLETION_TOKENS}, "
            f"total_tokens={total_tokens}, requests={REQUEST_COUNT}"
        )
        print(
            f"[EFFICIENCY] avg_prompt_tokens_per_req={TOTAL_PROMPT_TOKENS / REQUEST_COUNT:.1f}, "
            f"avg_completion_tokens_per_req={TOTAL_COMPLETION_TOKENS / REQUEST_COUNT:.1f}, "
            f"avg_total_tokens_per_req={total_tokens / REQUEST_COUNT:.1f}"
        )
        print(
            f"[COST] input_tokens_k={total_prompt_k:.2f}, output_tokens_k={total_completion_k:.2f}, "
            f"estimated_input_cost=${cost_input:.4f}, estimated_output_cost=${cost_output:.4f}, "
            f"estimated_total_cost=${total_cost:.4f}"
        )

    print("\nEvaluation completed.")
    print(f"Results saved to: {OUTPUT_JSON_PATH}")
    print(f"Score table saved to: {OUTPUT_MD_PATH}")





