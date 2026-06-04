from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional
import time
import threading

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -------------------- Path Config --------------------
INPUT_SUGGESTIONS_JSON = Path("")
INPUT_WRITING_BEHAVIOR_JSON = Path("")
INPUT_SEGMENT_CLASSIFICATION_JSON = Path("")
INPUT_EXPECTED_CI_JSON = Path("")

DEFAULT_OUTPUT_JSON = Path("")
DEFAULT_OUTPUT_MD = Path("")

MODEL = "openai/gpt-5.1"
TIMEOUT = 120
TEMPERATURE = 0.1
MAX_WORKERS = 20
PROMPT_COST_PER_1M = 3.0
COMPLETION_COST_PER_1M = 15.0
COUNTER_LOCK = threading.Lock()

# -------------------- Helper Functions --------------------
def build_client() -> OpenAI:
    if OpenAI is None:
        raise ImportError("Missing dependency: openai. Install with `pip install openai`.")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is missing. Please set OPENROUTER_API_KEY or OPENAI_API_KEY.")
    
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
    return OpenAI(api_key=api_key.strip(), base_url=base_url, timeout=TIMEOUT)

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def try_load_segment_text_map(source_segments_path: str) -> Dict[int, str]:
    if not source_segments_path:
        return {}
    payload = load_json(Path(source_segments_path))
    return {int(seg["knowledge_point_id"]): str(seg.get("text", "")).strip() 
            for seg in payload.get("segments", []) if "knowledge_point_id" in seg}

def build_kp_map(payload: Dict[str, Any], key_name: str = "results") -> Dict[int, Dict[str, Any]]:
    rows = payload.get(key_name, payload.get("knowledge_point_results", []))
    return {int(row["knowledge_point_id"]): row for row in rows if isinstance(row, dict) and "knowledge_point_id" in row}

def build_minute_counts(payload: Dict[str, Any]) -> Dict[int, int]:
    counts = {}
    for hit in payload.get("hits", []):
        if isinstance(hit, dict) and isinstance(hit.get("minute"), int):
            m = hit["minute"]
            counts[m] = counts.get(m, 0) + 1
    return counts

def build_a_feature_map(kp_map: Dict[int, Dict[str, Any]], minute_counts: Dict[int, int]) -> Dict[int, Dict[str, Any]]:
    if not kp_map:
        return {}
    max_minute = max(minute_counts.keys()) if minute_counts else 59
    max_end_char = max(int(v.get("end_char", 0) or 0) for v in kp_map.values()) or 1
    out = {}
    for kp_id, row in kp_map.items():
        s_m = max(0, min(max_minute, int(round((int(row.get("start_char", 0) or 0) / max_end_char) * max_minute))))
        e_m = max(0, min(max_minute, int(round((int(row.get("end_char", 0) or 0) / max_end_char) * max_minute))))
        mins = list(range(s_m, max(s_m, e_m) + 1))
        counts = [minute_counts.get(m, 0) for m in mins] or [0]
        out[kp_id] = {
            "start_minute": s_m, "end_minute": e_m, "window_len_min": len(mins),
            "a_total_hits": sum(counts), "a_avg_hits_per_minute": round(mean(counts), 3), "a_peak_hits_per_minute": max(counts)
        }
    return out

# -------------------- Judge Prompt Builder --------------------
def build_judge_prompt(item: Dict[str, Any], segment_text: str, extra_context: Dict[str, Any]) -> str:
    return f"""You are a senior educational evaluation expert. Please objectively and rationally evaluate the following "teaching improvement suggestions (ICAP Suggestions)" tailored for a specific knowledge point.

[Core Field Definitions]
1. gap_diagnosis: Explain why coverage is bottlenecked and if behavior converted into outputs. Quote transcript.
2. proposed_rewrite: Concept-level pedagogical modification to transition into target mode.
3. scaffolding_strategy: 1-2 highly operational strategies. Align tightly with the specific constraints for Mode C or Mode I specified in system instructions.
4. new_question: A newly designed prompt or question to stimulate the target cognitive mode.
5. reference_answer: A high-quality reference answer corresponding to the new question, aligned with subject-matter logic.

[Three Major Evaluation Dimensions and Scoring Standards (1-5 points)]
A) Accuracy:
   - Evaluate whether the diagnosis is accurate based on standard ICAP definitions (Interactive > Constructive > Active > Passive).
   - Check whether the suggested evolution direction (from current_ci to target_mode) is pedagogically logical.
B) Alignment & Grounding:
   - Goal Alignment: Whether the suggestion content effectively serves to achieve the `target_mode`.
   - Grounding Alignment (CRITICAL): Whether the historical classroom facts referenced in the suggestion (e.g., speaker counts, metrics) are consistent with the provided Extra Context.
   - [Exemption Policy]: `new_question`, `proposed_rewrite`, and `reference_answer` are future-oriented teaching design innovations. As long as they fit the current knowledge point topic and conform to pedagogical logic, they MUST NOT be flagged as hallucinations or penalized simply because the exact phrasing or question did not appear verbatim in the original classroom transcript.
C) Actionability:
   - Judge by "Balanced Practicality": suggestions should be reasonable, targeted, and executable.
   - Must explicitly specify: dominant actor (teacher/students), concrete action verbs, and when to apply in class flow.
   - Penalize BOTH extremes:
     (a) Too vague: slogan-like statements (e.g., "improve interaction") with no concrete moves.
     (b) Too detailed: over-scripted micromanagement with excessive minute-by-minute choreography that is unrealistic for real teaching.
   - Best Actionability is "middle granularity":
     clear and specific enough to execute, but still flexible for teacher adaptation.
   - Topic relevance is required: actions should map to this knowledge-point content, not generic reusable templates only.

[Actionability scoring anchor]
- 5: Strongly targeted + clear actor/action/timing + practical classroom fit + not over-detailed.
- 4: Good and executable, minor specificity/timing gaps.
- 3: Partly executable but either somewhat generic or somewhat over-engineered.
- 2: Mostly vague or impractically over-detailed; weak classroom fit.
- 1: Not actionable.

[Scoring Constraints]
- overall_score_5 = round((accuracy + alignment_grounding + actionability) / 3, 2)
- verdict: pass (>=3.75), borderline (3.00-3.74), fail (<3.00)

[Exhaustive Issue Mining Requirement]
- You must inspect ALL five suggestion fields (`gap_diagnosis`, `proposed_rewrite`, `scaffolding_strategy`, `new_question`, `reference_answer`).
- You must cross-check ALL trusted context sources (expected policy, classification summary, A-behavior features, transcript evidence).
- You must list every concrete defect you find in `major_issues` (no vague wording).
- If there are grounding inconsistencies, include them in both `hallucination_flags` and `major_issues`.
- If no issue is found for a field, explicitly state why in that field's comment.

Please output strictly in the following JSON format, without any markdown wrapper or extra explanations:
{{
  "knowledge_point_id": {item.get("knowledge_point_id")},
  "topic": "{str(item.get('topic', '')).replace('"', '\\"')}",
  "rederived_ci_level": "c|i|a|p|unknown",
  "scores": {{ "accuracy": 5, "alignment_grounding": 5, "actionability": 5 }},
  "field_quality": {{
    "gap_diagnosis": {{"score": 5, "comment": "..."}},
    "proposed_rewrite": {{"score": 5, "comment": "..."}},
    "scaffolding_strategy": {{"score": 5, "comment": "..."}},
    "new_question": {{"score": 5, "comment": "..."}},
    "reference_answer": {{"score": 5, "comment": "..."}}
  }},
  "consistency_checks": {{
    "teacher_current_ci": "{str(item.get('teacher_current_ci', '')).lower()}",
    "target_mode": "{str(item.get('target_mode', '')).lower()}",
    "is_teacher_ci_consistent_with_rederived": true,
    "is_suggestion_aligned_to_target_mode": true
  }},
  "hallucination_flags": [],
  "major_issues": [],
  "improvement_suggestions": [],
  "dimension_rationales": {{ "accuracy": "...", "alignment_grounding": "...", "actionability": "..." }},
  "overall_score_5": 5.0,
  "verdict": "pass"
}}

[Data Item to be Evaluated]
{json.dumps(item, ensure_ascii=False, indent=2)}

[Multi-source Pipeline Context Data (Trusted)]
{json.dumps(extra_context, ensure_ascii=False, indent=2)}

[Original Classroom Transcript for This Knowledge Point (For Grounding Comparison)]
{segment_text}
"""

# -------------------- Execution Core --------------------
def call_judge(client: OpenAI, item: Dict[str, Any], segment_text: str, extra_context: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_judge_prompt(item, segment_text, extra_context)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    content = (response.choices[0].message.content or "{}").strip()
    if content.startswith("```"):
        content = re.sub(r"^```json\s*|```$", "", content, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        content = m.group(0)
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    estimated_cost_usd = (
        (prompt_tokens / 1_000_000.0) * PROMPT_COST_PER_1M
        + (completion_tokens / 1_000_000.0) * COMPLETION_COST_PER_1M
    )
    return {
        "result": json.loads(content),
        "efficiency": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
        },
    }

def _extract_quoted_spans(text: str) -> List[str]:
    if not text:
        return []
    spans = re.findall(r"[\"“”'‘’]([^\"“”'‘’]{6,180})[\"“”'‘’]", text)
    return [s.strip() for s in spans if s and s.strip()]

def _has_verifiable_quote_in_transcript(gap_diagnosis: str, transcript: str) -> bool:
    transcript_l = (transcript or "").lower()
    if not transcript_l:
        return False
    spans = _extract_quoted_spans(gap_diagnosis or "")
    for s in spans:
        if s.lower() in transcript_l:
            return True
    return False

def _is_actionable_enough(scaffolding_strategy: str) -> bool:
    text = (scaffolding_strategy or "").lower()
    if not text:
        return False
    has_actor = any(k in text for k in ["teacher", "teachers", "learner", "learners", "student", "students", "教师", "学生"])
    has_action = any(
        k in text
        for k in [
            "ask", "prompt", "write", "discuss", "assign", "provide", "collect", "summarize",
            "debate", "reflect", "organize", "rotate", "evaluate", "feedback",
            "引导", "提问", "书写", "讨论", "组织", "分配", "反馈", "总结"
        ]
    )
    has_timing = any(k in text for k in ["after", "before", "during", "each", "weekly", "minute", "课前", "课中", "课后", "每次", "之后"])
    return has_actor and has_action and has_timing

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default

def _append_unique(lst: List[str], msg: str) -> None:
    if msg not in lst:
        lst.append(msg)

def apply_verifiable_caps(
    result: Dict[str, Any],
    item: Dict[str, Any],
    segment_text: str,
    extra_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Minimal but strict hard-check layer:
    - No verifiable quote in gap_diagnosis => accuracy capped at 3.
    - Scaffolding missing actor+action+timing => actionability capped at 3.
    - Target-mode constraint mismatch (C/I) => alignment_grounding capped at 3.
    """
    scores = result.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}

    flags = list(result.get("hallucination_flags", []) or [])
    issues = list(result.get("major_issues", []) or [])

    gap_text = str(item.get("gap_diagnosis", "") or "")
    scaffold_text = str(item.get("scaffolding_strategy", "") or "")
    new_q = str(item.get("new_question", "") or "")
    target_mode = str(item.get("target_mode", "") or "").strip().lower()

    # 1) Evidence-verifiable cap for accuracy
    if not _has_verifiable_quote_in_transcript(gap_text, segment_text):
        scores["accuracy"] = min(_safe_float(scores.get("accuracy"), 0.0), 3.0)
        _append_unique(flags, "gap_diagnosis_no_verifiable_quote")
        _append_unique(issues, "gap_diagnosis lacks verifiable quoted evidence that can be matched in transcript.")

    # 2) Executability cap for actionability
    if not _is_actionable_enough(scaffold_text):
        scores["actionability"] = min(_safe_float(scores.get("actionability"), 0.0), 3.0)
        _append_unique(flags, "scaffolding_missing_actor_action_timing")
        _append_unique(issues, "scaffolding_strategy misses one or more of actor/action/timing elements.")

    # 3) Target-mode hard constraints (use item + context)
    lower_scaffold = scaffold_text.lower()
    lower_new_q = new_q.lower()
    merged_text = f"{lower_scaffold}\n{lower_new_q}"
    if target_mode == "c":
        forbidden_peer_words = [
            "group", "peer", "partner", "discuss", "collaborate", "team", "talk to",
            "小组", "同伴", "结对", "合作", "讨论"
        ]
        if any(w in merged_text for w in forbidden_peer_words):
            scores["alignment_grounding"] = min(_safe_float(scores.get("alignment_grounding"), 0.0), 3.0)
            _append_unique(flags, "mode_c_contains_peer_interaction_design")
            _append_unique(issues, "target_mode=C but strategy/question includes peer interaction signals.")
    elif target_mode == "i":
        interactive_words = [
            "discuss", "debate", "peer", "partner", "collaborate", "co-construct", "exchange",
            "讨论", "辩论", "同伴", "合作", "共建"
        ]
        if not any(w in merged_text for w in interactive_words):
            scores["alignment_grounding"] = min(_safe_float(scores.get("alignment_grounding"), 0.0), 3.0)
            _append_unique(flags, "mode_i_missing_interaction_design")
            _append_unique(issues, "target_mode=I but strategy/question lacks explicit interaction design.")

    # 4) CI consistency check using available context
    teacher_current_ci = str(item.get("teacher_current_ci", "") or "").strip().lower()
    rederived_ci = str(result.get("rederived_ci_level", "") or "").strip().lower()
    if teacher_current_ci and rederived_ci and rederived_ci != "unknown" and teacher_current_ci != rederived_ci:
        _append_unique(flags, "teacher_ci_rederived_mismatch")
        _append_unique(issues, f"teacher_current_ci ({teacher_current_ci}) conflicts with rederived_ci_level ({rederived_ci}).")

    # Recompute overall and verdict with unchanged policy
    acc = _safe_float(scores.get("accuracy"), 0.0)
    align = _safe_float(scores.get("alignment_grounding"), 0.0)
    act = _safe_float(scores.get("actionability"), 0.0)
    overall = round((acc + align + act) / 3.0, 2)
    if overall >= 3.75:
        verdict = "pass"
    elif overall >= 3.0:
        verdict = "borderline"
    else:
        verdict = "fail"

    result["scores"] = scores
    result["hallucination_flags"] = flags
    result["major_issues"] = issues
    result["overall_score_5"] = overall
    result["verdict"] = verdict
    return result

def evaluate_one(
    client: OpenAI, item: Dict[str, Any], segment_text_map: Dict[int, str],
    class_kp_map: Dict[int, Dict[str, Any]],
    a_feature_map: Dict[int, Dict[str, Any]], expected_kp_map: Dict[int, Dict[str, Any]],
    expected_global_config: Dict[str, Any]
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        kp_id = int(item.get("knowledge_point_id", -1))
    except (TypeError, ValueError):
        kp_id = -1

    seg_text = segment_text_map.get(kp_id, "") or str(class_kp_map.get(kp_id, {}).get("text", "")).strip()
    if not seg_text: seg_text = "[No segment text available for grounding.]"

    cls_row = class_kp_map.get(kp_id, {})
    cls_info = cls_row.get("classification", {}) if isinstance(cls_row, dict) else {}
    speaker_ev = cls_info.get("speaker_evidence", {}) if isinstance(cls_info, dict) else {}

    extra_context = {
        "knowledge_point_id": kp_id,
        "expected_policy_global_config": expected_global_config,
        "expected_policy_for_this_kp": expected_kp_map.get(kp_id, {}),
        "classification_summary": {
            "student_current_ci": cls_info.get("final_ci_level"),
            "student_speaker_count": cls_info.get("student_speaker_count"),
            "student_current_reason": cls_info.get("final_ci_reason"),
            "teacher_examples": speaker_ev.get("teacher_examples", []) if isinstance(speaker_ev, dict) else [],
            "student_examples": speaker_ev.get("student_examples", []) if isinstance(speaker_ev, dict) else [],
        },
        "a_behavior_features": a_feature_map.get(kp_id, {}),
    }

    try:
        judge = call_judge(client, item, seg_text, extra_context)
        res = judge["result"]
        res.update({"knowledge_point_id": item.get("knowledge_point_id"), "topic": item.get("topic", "")})
        res = apply_verifiable_caps(res, item, seg_text, extra_context)
        res["efficiency"] = {
            **judge["efficiency"],
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        return res
    except Exception as e:
        return {
            "knowledge_point_id": kp_id,
            "topic": item.get("topic", ""),
            "verdict": "fail",
            "major_issues": [f"Error: {e}"],
            "efficiency": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        }

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"count": 0}

    def get_score(r, k1, k2=None):
        v = r.get(k1, {})
        raw = v.get(k2) if k2 else v
        try:
            return float(raw if raw is not None else 0.0)
        except (TypeError, ValueError):
            return 0.0
    
    accs = [get_score(r, "scores", "accuracy") for r in results]
    aligns = [get_score(r, "scores", "alignment_grounding") for r in results]
    acts = [get_score(r, "scores", "actionability") for r in results]
    overalls = [float(r.get("overall_score_5", 0)) for r in results]
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
        "# ICAP Suggestions Evaluation Report",
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
        "| KP | Topic | Accuracy | Alignment & Grounding | Actionability | Total(1-5) | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in rows:
        scores = item.get("scores", {})
        lines.append(
            f"| {item.get('knowledge_point_id')} | {str(item.get('topic','')).replace('|','/')} | "
            f"{scores.get('accuracy')} | {scores.get('alignment_grounding')} | {scores.get('actionability')} | "
            f"{item.get('overall_score_5')} | {item.get('verdict')} |"
        )
    lines.append(
        f"| **average** | - | **{agg.get('avg_accuracy')}** | **{agg.get('avg_alignment_grounding')}** | "
        f"**{agg.get('avg_actionability')}** | **{agg.get('avg_overall_score_5')}** | - |"
    )
    lines.append(
        f"| **sum** | - | **{round(sum_acc,3)}** | **{round(sum_align,3)}** | "
        f"**{round(sum_act,3)}** | **{round(sum_total,3)}** | - |"
    )
    return "\n".join(lines)

def main() -> None:
    run_start = time.perf_counter()
    client = build_client()
    sug_payload = load_json(INPUT_SUGGESTIONS_JSON)
    items = sug_payload.get("results", [])
    if not isinstance(items, list): raise ValueError("Invalid evaluation data: 'results' must be a list.")

    segment_text_map = try_load_segment_text_map(sug_payload.get("source_knowledge_segments", ""))
    class_kp_map = build_kp_map(load_json(INPUT_SEGMENT_CLASSIFICATION_JSON))
    expected_payload = load_json(INPUT_EXPECTED_CI_JSON)
    expected_kp_map = build_kp_map(expected_payload)
    
    expected_global_config = {
        "class_size": expected_payload.get("class_size"),
        "c_ratio_threshold": expected_payload.get("c_ratio_threshold"),
        "summary": expected_payload.get("summary", {})
    }
    a_feature_map = build_a_feature_map(class_kp_map, build_minute_counts(load_json(INPUT_WRITING_BEHAVIOR_JSON)))

    print(f"Starting parallel evaluation. Total items: {len(items)}")
    ordered_results = [None] * len(items)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(
                evaluate_one, client, item, segment_text_map,
                class_kp_map, a_feature_map, expected_kp_map, expected_global_config
            ): idx for idx, item in enumerate(items)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            ordered_results[idx] = future.result()

    valid_results = [r for r in ordered_results if r is not None]
    total_prompt_tokens = sum(int(r.get("efficiency", {}).get("prompt_tokens", 0)) for r in valid_results)
    total_completion_tokens = sum(int(r.get("efficiency", {}).get("completion_tokens", 0)) for r in valid_results)
    total_tokens = sum(int(r.get("efficiency", {}).get("total_tokens", 0)) for r in valid_results)
    total_cost_usd = round(sum(float(r.get("efficiency", {}).get("estimated_cost_usd", 0.0)) for r in valid_results), 6)
    total_elapsed_ms = sum(int(r.get("efficiency", {}).get("elapsed_ms", 0)) for r in valid_results)
    elapsed_sec = round(time.perf_counter() - run_start, 3)
    report = {
        "model": MODEL,
        "evaluation_dimensions": ["Accuracy", "Alignment & Grounding", "Actionability"],
        "aggregate": aggregate(valid_results),
        "efficiency": {
            "elapsed_sec": elapsed_sec,
            "request_count": len(valid_results),
            "pricing_model_reference": MODEL,
            "prompt_cost_per_1m": PROMPT_COST_PER_1M,
            "completion_cost_per_1m": COMPLETION_COST_PER_1M,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "llm_elapsed_ms_sum": total_elapsed_ms,
            "estimated_total_cost_usd": total_cost_usd,
        },
        "results": valid_results,
    }

    DEFAULT_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    DEFAULT_OUTPUT_MD.write_text(to_markdown(report), encoding="utf-8")
    print("Evaluation completed successfully.")
    print(
        f"[USAGE] elapsed={elapsed_sec}s, tokens={total_tokens} "
        f"(prompt={total_prompt_tokens}, completion={total_completion_tokens}), "
        f"cost≈${total_cost_usd}"
    )

if __name__ == "__main__":
    main()
