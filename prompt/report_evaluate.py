import os
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from openai import OpenAI

_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_openai_key = os.getenv("OPENAI_API_KEY")

if _openrouter_key and _openrouter_key.strip():
    # Use OpenRouter
    MODEL = "openai/gpt-5.1"
    client = OpenAI(
        api_key=_openrouter_key.strip(),
        base_url="https://openrouter.ai/api/v1"
    )
elif _openai_key and _openai_key.strip():
    # Use official OpenAI API
    MODEL = "gpt-4o-mini"
    client = OpenAI(api_key=_openai_key.strip())
else:
    raise ValueError(
        "API key is not set. Please configure one of the following:\n"
        "  - set OPENROUTER_API_KEY=your_key   (OpenRouter)\n"
        "  - set OPENAI_API_KEY=your_key       (Official OpenAI API)"
    )
TIMEOUT = 60
TEMPERATURE = 0  # Evaluation must use temperature 0
MAX_WORKERS = max(1, int(os.getenv("PROMPT_EVAL_MAX_WORKERS", "20")))

# Prompt-only big JSON input (contains all knowledge points)
TEACHING_REPORTS_PATH = r""
# Knowledge segments JSON (used to provide transcript context per knowledge point)
TRANSCRIPT_SEGMENTS_JSON = r""
    # Evaluation output path (single file containing all window results)
OUTPUT_JSON_PATH = r""
OUTPUT_MD_PATH = r""

# Token and cost statistics (prices can be updated to latest OpenAI pricing)
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0
PRICE_INPUT_PER_1K = 0.00015   # USD per 1K prompt tokens (gpt-4o-mini)
PRICE_OUTPUT_PER_1K = 0.00060  # USD per 1K completion tokens
COUNTER_LOCK = threading.Lock()

# One-pass evaluator prompt (no dean/calibration stage)
EVALUATOR_PROMPT_TEMPLATE = """
You are evaluating ONE feedback artifact: classroom teaching suggestions.
The goal is to judge feedback quality, not to judge transcript content itself.

[Object Being Evaluated]
- teaching_suggestions: critique/instruction text intended to guide teaching improvement.

[ICAP Reference]
- Standard ICAP hierarchy: Interactive > Constructive > Active > Passive.
- For this task, use ICAP as pedagogical grounding: feedback should not contradict the intended cognitive-engagement direction implied by the critique context.

[Evaluation Dimensions]
A) Accuracy:
   - Judge whether teaching_suggestions correctly identify concrete problems in the provided context and do not distort what is present.
   - Check whether pedagogical direction is logically consistent with ICAP-informed improvement (at least not moving in a contradictory direction).
   - Penalize wrong diagnosis, unsupported claims, or ICAP-inconsistent direction.
   - Be conservative: if diagnosis is generic or weakly evidenced, score cannot exceed 3.

B) Alignment & Grounding:
   - Goal Alignment: check whether teaching_suggestions are aligned with the optimization objective:
     clarity/structure, pacing, transitions/signposting, learner engagement, and factual preservation.
   - Grounding Alignment (CRITICAL): check whether claims are grounded in the provided suggestions and transcript context (no invented references).
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

[Suggestions To Evaluate]
{feedback}

[Transcript Context]
{transcript_text}

Output strictly JSON:
{{
  "scores": {{ "accuracy": 5, "alignment_grounding": 5, "actionability": 5 }},
  "dimension_rationales": {{ "accuracy": "...", "alignment_grounding": "...", "actionability": "..." }},
  "major_issues": [],
  "improvement_suggestions": [],
  "overall_score_5": 5.0,
  "verdict": "pass"
}}
"""



def safe_json_loads(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Empty LLM output")

    # Extract the first JSON object (automatically strips ```json ... ``` wrappers)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")

    json_str = match.group(0)
    return json.loads(json_str)

# LLM call function

def call_llm(prompt: str) -> str:
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
            TOTAL_PROMPT_TOKENS += getattr(usage, "prompt_tokens", 0)
            TOTAL_COMPLETION_TOKENS += getattr(usage, "completion_tokens", 0)
        REQUEST_COUNT += 1
    return response.choices[0].message.content


def load_prompt_results(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt-only result JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("Invalid prompt-only JSON: `results` must be a list.")
    return [r for r in rows if isinstance(r, dict)]


def load_transcript_map(path: str) -> Dict[int, str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Knowledge segments JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("Invalid segments JSON: `segments` must be a list.")
    out: Dict[int, str] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        kp_id = seg.get("knowledge_point_id")
        if isinstance(kp_id, int):
            out[kp_id] = str(seg.get("text", "")).strip()
    return out


def save_evaluation_result(output_path: str, result: Dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def evaluate_feedback(feedback: str, transcript: str = "", engagement_evidence: str = "") -> Dict:
    _ = engagement_evidence  # kept for compatibility
    prompt = EVALUATOR_PROMPT_TEMPLATE.format(
        feedback=feedback,
        transcript_text=(transcript[:8000] if transcript and transcript.strip() else "[No transcript provided]"),
    )
    raw_output = call_llm(prompt)
    result = safe_json_loads(raw_output)
    return result


def evaluate_item(item: Dict[str, Any], transcript_map: Dict[int, str]) -> Dict:
    kp_id = item.get("knowledge_point_id")
    if not isinstance(kp_id, int):
        return {"label": "unknown", "result": {"error": "Missing or invalid knowledge_point_id"}}

    label = f"kp_{kp_id}"
    suggestions = item.get("teaching_suggestions", [])
    if isinstance(suggestions, list):
        feedback_text = "\n\n".join(str(s) for s in suggestions)
    else:
        feedback_text = str(suggestions)
    transcript_text = transcript_map.get(kp_id, "")

    try:
        engagement_evidence = ""
        if not feedback_text.strip():
            return {"label": label, "result": {"error": "Empty teaching_suggestions"}}

        result = evaluate_feedback(
            feedback_text,
            transcript=transcript_text,
            engagement_evidence=engagement_evidence or "",
        )
        result["knowledge_point_id"] = kp_id
        result["topic"] = str(item.get("topic", ""))
        return {"label": label, "result": result}
    except Exception as e:
        return {"label": label, "result": {"error": str(e)}}


def to_markdown_table(all_results: Dict[str, Dict[str, Any]]) -> str:
    rows: List[Dict[str, Any]] = []
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
        rows.append(
            {
                "label": label,
                "kp_id": result.get("knowledge_point_id", ""),
                "topic": str(result.get("topic", "")).replace("|", "/"),
                "accuracy": accuracy,
                "alignment_grounding": alignment,
                "actionability": actionability,
                "overall_score_5": overall,
                "verdict": str(result.get("verdict", "")),
            }
        )

    rows.sort(key=lambda x: str(x.get("label", "")))
    if not rows:
        return "# Prompt-Only Evaluation Table\n\nNo valid rows."

    avg_accuracy = sum(r["accuracy"] for r in rows) / len(rows)
    avg_alignment = sum(r["alignment_grounding"] for r in rows) / len(rows)
    avg_actionability = sum(r["actionability"] for r in rows) / len(rows)
    avg_overall = sum(r["overall_score_5"] for r in rows) / len(rows)

    lines = [
        "# Prompt-Only Evaluation Table",
        "",
        "| Group | KP | Topic | Accuracy | Alignment & Grounding | Actionability | Overall(1-5) | Verdict |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['kp_id']} | {row['topic']} | "
            f"{row['accuracy']:.2f} | {row['alignment_grounding']:.2f} | {row['actionability']:.2f} | "
            f"{row['overall_score_5']:.2f} | {row['verdict']} |"
        )
    lines.append(
        f"| **平均值** | - | - | **{avg_accuracy:.2f}** | **{avg_alignment:.2f}** | "
        f"**{avg_actionability:.2f}** | **{avg_overall:.2f}** | - |"
    )
    return "\n".join(lines)




# Entry point

if __name__ == "__main__":
    try:
        items = load_prompt_results(TEACHING_REPORTS_PATH)
        transcript_map = load_transcript_map(TRANSCRIPT_SEGMENTS_JSON)
    except Exception as e:
        print(f"[ERROR] Failed to load inputs: {e}")
        exit(1)
    if not items:
        print(f"[ERROR] No results found in {TEACHING_REPORTS_PATH}")
        exit(1)

    all_results = {}
    print(f"[START] knowledge_points={len(items)}, workers={MAX_WORKERS}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_label = {executor.submit(evaluate_item, item, transcript_map): f"kp_{item.get('knowledge_point_id')}" for item in items}
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            print(f"\n[EVAL] {label}")
            try:
                payload = future.result()
                label = payload.get("label", label)
                result = payload.get("result", {})
                all_results[label] = result
                if "error" in result:
                    print(f"  [SKIP] {result['error']}")
                else:
                    score = result.get("overall_score_5", "?")
                    verdict = result.get("verdict", "?")
                    print(f"  [DONE] overall_score_5={score}, verdict={verdict}")
            except Exception as e:
                print(f"  [FAIL] {e}")
                all_results[label] = {"error": str(e)}

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    table_md = to_markdown_table(all_results)
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(table_md)

    # Token usage, efficiency, and cost
    if REQUEST_COUNT > 0:
        total_prompt_k = TOTAL_PROMPT_TOKENS / 1000.0
        total_completion_k = TOTAL_COMPLETION_TOKENS / 1000.0
        cost_input = total_prompt_k * PRICE_INPUT_PER_1K
        cost_output = total_completion_k * PRICE_OUTPUT_PER_1K
        total_cost = cost_input + cost_output
        avg_prompt = TOTAL_PROMPT_TOKENS / REQUEST_COUNT
        avg_completion = TOTAL_COMPLETION_TOKENS / REQUEST_COUNT
        total_tokens = TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS
        avg_total_per_req = total_tokens / REQUEST_COUNT
        print(
            f"\n[TOKEN] total_prompt_tokens={TOTAL_PROMPT_TOKENS}, "
            f"total_completion_tokens={TOTAL_COMPLETION_TOKENS}, "
            f"total_tokens={total_tokens}, "
            f"requests={REQUEST_COUNT}"
        )
        print(
            f"[EFFICIENCY] avg_prompt_tokens_per_req={avg_prompt:.1f}, "
            f"avg_completion_tokens_per_req={avg_completion:.1f}, "
            f"avg_total_tokens_per_req={avg_total_per_req:.1f}"
        )
        print(
            f"[COST] input_tokens_k={total_prompt_k:.2f}, "
            f"output_tokens_k={total_completion_k:.2f}, "
            f"estimated_input_cost=${cost_input:.4f}, "
            f"estimated_output_cost=${cost_output:.4f}, "
            f"estimated_total_cost=${total_cost:.4f}"
        )

    print("\nEvaluation completed.")
    print(f"Results saved to: {OUTPUT_JSON_PATH}")
    print(f"Score table saved to: {OUTPUT_MD_PATH}")





