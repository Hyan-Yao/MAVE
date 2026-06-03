
# [TOKEN] prompt=12086, completion=1956, total=14042, requests=10
# [EFFICIENCY] elapsed=7.372s, avg_total_tokens_per_req=1404.2, throughput_tokens_per_sec=1904.9
# [COST] input_tokens_k=12.09, output_tokens_k=1.96, estimated_input_cost=$0.0018, estimated_output_cost=$0.0012, estimated_total_cost=$0.0030
# [TOKEN] prompt=8872, completion=1347, total=10219, requests=7

# [EFFICIENCY] elapsed=10.133s, avg_total_tokens_per_req=1459.9, throughput_tokens_per_sec=1008.5
# [COST] input_tokens_k=8.87, output_tokens_k=1.35, estimated_input_cost=$0.0013, estimated_output_cost=$0.0008, estimated_total_cost=$0.0021]

# [TOKEN] prompt=8063, completion=1152, total=9215, requests=7
# [EFFICIENCY] elapsed=15.072s, avg_total_tokens_per_req=1316.4, throughput_tokens_per_sec=611.4
# [COST] input_tokens_k=8.06, output_tokens_k=1.15, estimated_input_cost=$0.0012, estimated_output_cost=$0.0007, estimated_total_cost=$0.0019

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI


# API config (consistent with other project scripts: prefer OpenRouter, fallback to official OpenAI)
_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_openai_key = os.getenv("OPENAI_API_KEY")

if _openrouter_key and _openrouter_key.strip():
    MODEL = "openai/gpt-4o-mini"
    client = OpenAI(
        api_key=_openrouter_key.strip(),
        base_url="https://openrouter.ai/api/v1",
    )
elif _openai_key and _openai_key.strip():
    MODEL = "gpt-4o-mini"
    client = OpenAI(api_key=_openai_key.strip())
else:
    raise ValueError(
        "API key is not set. Please configure one of the following:\n"
        "  - set OPENROUTER_API_KEY=your_key   (OpenRouter)\n"
        "  - set OPENAI_API_KEY=your_key       (Official OpenAI API)"
    )


# Path configuration (knowledge-point based input)
KNOWLEDGE_SEGMENTS_JSON = (
    r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Meeting for 3D Design/icap_output/icap_split_free/knowledge_segments_global_free.json"
)
OUTPUT_DIR = (
    r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/prompt/now_stem/teaching_suggestions_prompt_only_by_kp"
)
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "teaching_suggestions_prompt_only_by_kp.json")

# Inference parameters
TIMEOUT = int(os.getenv("PROMPT_ONLY_TIMEOUT", "120"))
TEMPERATURE = float(os.getenv("PROMPT_ONLY_TEMPERATURE", "0.2"))
MAX_WORKERS = max(1, int(os.getenv("PROMPT_ONLY_MAX_WORKERS", "20")))

# Token statistics
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0
PROMPT_COST_PER_1M = 0.15
COMPLETION_COST_PER_1M = 0.60
USAGE_LOCK = threading.Lock()

# Maximum suggestions kept per transcript (extra paragraphs are truncated)
MAX_SUGGESTIONS = 3

SYSTEM_PROMPT = """
Provide practical teaching suggestions based only on this knowledge-point transcript segment. At most 3.
"""


def parse_json_object_from_text(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if not content:
        raise ValueError("Empty model output.")
    if content.startswith("```"):
        content = re.sub(r"^```json\s*|```$", "", content, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        content = m.group(0)
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def predict_icap_state(kp_id: int, topic: str, transcript_text: str) -> Dict[str, str]:
    """
    Predict ICAP state with ordinal definition.
    Hard rule requested by user:
    - If no student speech is detected, mark as p/a.
    """
    sys_prompt = (
        "You are an ICAP classifier. This is an ORDINAL prediction task: p/a < c < i. "
        "Return JSON only."
    )
    user_prompt = (
        f"Knowledge Point ID: {kp_id}\n"
        f"Topic: {topic}\n\n"
        "Please classify the transcript by ICAP using these definitions:\n"
        "- p/a: no student speech detected, or only passive/attentive behavior without constructive output.\n"
        "- c: individual constructive output by students, but no explicit peer-to-peer co-construction.\n"
        "- i: explicit peer interaction/co-construction among students.\n\n"
        "Rules:\n"
        "1) This is ORDINAL prediction (p/a < c < i), not nominal.\n"
        "2) If student_speech_detected is false, expected_icap_level MUST be p/a.\n"
        "3) Output strict JSON only:\n"
        "{\n"
        '  "student_speech_detected": true,\n'
        '  "expected_icap_level": "p/a|c|i",\n'
        '  "reason": "short evidence-based reason"\n'
        "}\n\n"
        f"Transcript Segment:\n{transcript_text}"
    )
    raw = call_gpt(sys_prompt, user_prompt)
    obj = parse_json_object_from_text(raw)
    speech_detected = bool(obj.get("student_speech_detected", False))
    level = str(obj.get("expected_icap_level", "")).strip().lower()
    reason = str(obj.get("reason", "")).strip()

    if not speech_detected:
        level = "p/a"
        if not reason:
            reason = "No student speech detected."
    if level not in {"p/a", "c", "i"}:
        level = "c" if speech_detected else "p/a"
    if not reason:
        reason = "Predicted by ICAP ordinal classifier."
    return {
        "expected_icap_level": level,
        "expected_icap_reason": reason,
        "student_speech_detected": str(speech_detected).lower(),
    }


def parse_suggestions_from_text(text: str) -> Dict:
    if not text or not text.strip():
        raise ValueError("Empty model output.")
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not parts:
        raise ValueError("No suggestions parsed from model output.")
    return {"teaching_suggestions": parts[:MAX_SUGGESTIONS]}


def call_gpt(system_prompt: str, user_prompt: str) -> str:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, REQUEST_COUNT

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
    )
    usage = getattr(response, "usage", None)
    with USAGE_LOCK:
        if usage is not None:
            TOTAL_PROMPT_TOKENS += int(getattr(usage, "prompt_tokens", 0) or 0)
            TOTAL_COMPLETION_TOKENS += int(getattr(usage, "completion_tokens", 0) or 0)
        REQUEST_COUNT += 1
    return response.choices[0].message.content


def load_knowledge_points(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Knowledge segments file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("Invalid knowledge segments format: `segments` must be a list.")
    rows: List[Dict[str, Any]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        kp_id = item.get("knowledge_point_id")
        if not isinstance(kp_id, int):
            continue
        rows.append(item)
    rows.sort(key=lambda x: int(x.get("knowledge_point_id", 10**9)))
    return rows


def build_user_prompt(
    kp_id: int,
    topic: str,
    transcript_text: str,
    expected_icap_level: str,
    expected_icap_reason: str,
) -> str:
    return (
        f"Knowledge Point ID: {kp_id}\n"
        f"Topic: {topic}\n"
        f"Expected ICAP level (ordinal target): {expected_icap_level}\n"
        f"Why: {expected_icap_reason}\n\n"
        "Suggestion constraints:\n"
        "- If expected ICAP is p/a: do not fabricate student speech.\n"
        "- If expected ICAP is c: prefer individual constructive prompts.\n"
        "- If expected ICAP is i: prefer explicit peer interaction and co-construction prompts.\n\n"
        f"Transcript Segment:\n{transcript_text}"
    )


def process_one_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    kp_id = int(seg.get("knowledge_point_id"))
    topic = str(seg.get("topic", "")).strip()
    transcript_text = str(seg.get("text", "")).strip()

    if not transcript_text:
        return {
            "knowledge_point_id": kp_id,
            "topic": topic,
            "source_segments_json": KNOWLEDGE_SEGMENTS_JSON,
            "method": "prompt_only",
            "model_name": MODEL,
            "teaching_suggestions": [],
            "error": "empty segment text",
        }

    icap_pred = predict_icap_state(kp_id, topic, transcript_text)
    user_prompt = build_user_prompt(
        kp_id,
        topic,
        transcript_text,
        icap_pred["expected_icap_level"],
        icap_pred["expected_icap_reason"],
    )
    raw = call_gpt(SYSTEM_PROMPT, user_prompt)
    result = parse_suggestions_from_text(raw)
    return {
        "knowledge_point_id": kp_id,
        "topic": topic,
        "source_segments_json": KNOWLEDGE_SEGMENTS_JSON,
        "method": "prompt_only",
        "model_name": MODEL,
        **icap_pred,
        **result,
    }


def main() -> None:
    run_start = time.time()
    segments = load_knowledge_points(KNOWLEDGE_SEGMENTS_JSON)
    if not segments:
        raise ValueError(f"No valid knowledge points found in: {KNOWLEDGE_SEGMENTS_JSON}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_records: List[Optional[Dict]] = [None] * len(segments)
    print(f"[START] segments={len(segments)}, workers={MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_one_segment, seg): idx
            for idx, seg in enumerate(segments)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            seg = segments[idx]
            kp_id = int(seg.get("knowledge_point_id"))
            try:
                record = future.result()
                all_records[idx] = record
                if record.get("error"):
                    print(f"  [SKIP] knowledge_point_{kp_id}: {record.get('error')}")
                else:
                    print(f"  [OK] knowledge_point_{kp_id}")
            except Exception as e:
                all_records[idx] = {
                    "knowledge_point_id": kp_id,
                    "topic": str(seg.get("topic", "")).strip(),
                    "source_segments_json": KNOWLEDGE_SEGMENTS_JSON,
                    "method": "prompt_only",
                    "model_name": MODEL,
                    "teaching_suggestions": [],
                    "error": str(e),
                }
                print(f"  [FAIL] knowledge_point_{kp_id}: {e}")

    finalized_records = [r for r in all_records if isinstance(r, dict)]

    total_tokens = TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS
    elapsed_sec = max(0.001, time.time() - run_start)
    prompt_cost = (TOTAL_PROMPT_TOKENS / 1_000_000.0) * PROMPT_COST_PER_1M
    completion_cost = (TOTAL_COMPLETION_TOKENS / 1_000_000.0) * COMPLETION_COST_PER_1M
    total_cost = prompt_cost + completion_cost
    throughput = total_tokens / elapsed_sec
    success_count = sum(1 for r in finalized_records if not r.get("error"))
    fail_count = len(finalized_records) - success_count

    output_payload = {
        "method": "prompt_only",
        "model_name": MODEL,
        "source_segments_json": KNOWLEDGE_SEGMENTS_JSON,
        "segment_count": len(segments),
        "success_count": success_count,
        "fail_count": fail_count,
        "efficiency": {
            "elapsed_sec": round(elapsed_sec, 3),
            "prompt_tokens": TOTAL_PROMPT_TOKENS,
            "completion_tokens": TOTAL_COMPLETION_TOKENS,
            "total_tokens": total_tokens,
            "requests": REQUEST_COUNT,
            "avg_total_tokens_per_req": round((total_tokens / REQUEST_COUNT), 3) if REQUEST_COUNT > 0 else 0.0,
            "throughput_tokens_per_sec": round(throughput, 3),
            "pricing_model_reference": "gpt-4o-mini",
            "prompt_cost_per_1m": PROMPT_COST_PER_1M,
            "completion_cost_per_1m": COMPLETION_COST_PER_1M,
            "estimated_input_cost_usd": round(prompt_cost, 6),
            "estimated_output_cost_usd": round(completion_cost, 6),
            "estimated_total_cost_usd": round(total_cost, 6),
        },
        "results": finalized_records,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] json saved to: {OUTPUT_JSON_PATH}")
    if REQUEST_COUNT > 0:
        print(
            f"[TOKEN] prompt={TOTAL_PROMPT_TOKENS}, completion={TOTAL_COMPLETION_TOKENS}, "
            f"total={total_tokens}, requests={REQUEST_COUNT}"
        )
        print(
            f"[EFFICIENCY] elapsed={elapsed_sec:.3f}s, avg_total_tokens_per_req={total_tokens / REQUEST_COUNT:.1f}, "
            f"throughput_tokens_per_sec={throughput:.1f}"
        )
        print(
            f"[COST] input_tokens_k={TOTAL_PROMPT_TOKENS/1000.0:.2f}, output_tokens_k={TOTAL_COMPLETION_TOKENS/1000.0:.2f}, "
            f"estimated_input_cost=${prompt_cost:.4f}, estimated_output_cost=${completion_cost:.4f}, "
            f"estimated_total_cost=${total_cost:.4f}"
        )


if __name__ == "__main__":
    main()
