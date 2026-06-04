
import os
import sys
import json
import time
import re
import threading
import contextvars
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# Use the local TextGrad repository code (not the installed site-packages version).
TEXTGRAD_REPO_ROOT = os.getenv("TEXTGRAD_REPO_ROOT", "").strip()
if TEXTGRAD_REPO_ROOT and os.path.isdir(TEXTGRAD_REPO_ROOT) and TEXTGRAD_REPO_ROOT not in sys.path:
    sys.path.insert(0, TEXTGRAD_REPO_ROOT)

import textgrad as tg
from textgrad import Variable
from textgrad.model import BlackboxLLM
from textgrad.engine import get_engine


INPUT_JSON_PATH = ""
OUTPUT_JSONL_PATH = os.getenv(
    "TEXTGRAD_OUTPUT_JSONL",
    "",
)

# By default, do one TextGrad step to limit cost. You can override with env var.
STEPS = int(os.getenv("TEXTGRAD_STEPS", "5"))

# To avoid runaway prompt size, you can cap transcript length (characters).
MAX_CHARS = int(os.getenv("TEXTGRAD_MAX_CHARS", "12000"))
MAX_WORKERS = max(1, int(os.getenv("TEXTGRAD_MAX_WORKERS", "20")))

# gpt-4o-mini price (USD per 1M tokens)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

_CURRENT_JOB = contextvars.ContextVar("textgrad_current_job", default="global")


def resolve_engine_and_model() -> Tuple[str, object]:
    """
    Resolve model routing to match focus_scoring_fine.py:
    - Prefer OpenRouter (OPENROUTER_API_KEY + base_url=https://openrouter.ai/api/v1)
    - Else fallback to OpenAI official (OPENAI_API_KEY)
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key and openrouter_key.strip():
        # For current textgrad version, OpenRouter should use experimental LiteLLM engine.
        os.environ["OPENROUTER_API_KEY"] = openrouter_key.strip()
        model_name = "experimental:openrouter/openai/gpt-4o-mini"
        engine = get_engine(model_name)
    elif openai_key and openai_key.strip():
        os.environ["OPENAI_API_KEY"] = openai_key.strip()
        model_name = "gpt-4o-mini"
        engine = get_engine(model_name)
    else:
        raise ValueError(
            "Missing API key. Please set either OPENROUTER_API_KEY or OPENAI_API_KEY."
        )
    return model_name, engine


def list_segments_from_json(json_path: str) -> List[Tuple[str, str, str]]:
    """
    Returns a list of (window_label, source_ref, transcript_text), sorted by knowledge_point_id.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"INPUT_JSON_PATH not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.loads(f.read())
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise RuntimeError(f"`segments` must be a list in: {json_path}")

    out: List[Tuple[str, str, str]] = []
    sorted_segments = sorted(
        [s for s in segments if isinstance(s, dict)],
        key=lambda s: int(s.get("knowledge_point_id", 0) or 0),
    )
    for idx, seg in enumerate(sorted_segments, start=1):
        kp_id = seg.get("knowledge_point_id", idx)
        topic = str(seg.get("topic", "")).strip()
        text = str(seg.get("text", "")).strip()
        window_label = f"kp_{kp_id}"
        source_ref = f"{json_path}#knowledge_point_id={kp_id}"
        if topic:
            window_label = f"{window_label}_{topic[:40]}"
        out.append((window_label, source_ref, text))
    return out


def normalize_model_for_pricing(model_name: str) -> str:
    m = (model_name or "").strip()
    if m.startswith("experimental:"):
        m = m.split("experimental:", 1)[1]
    if m.startswith("openrouter/"):
        m = m.split("openrouter/", 1)[1]
    return m


def estimate_tokens(model_name: str, text: str) -> int:
    """
    Prefer litellm token_counter when available; fallback to rough char-based estimate.
    """
    content = str(text or "")
    if not content:
        return 0
    try:
        from litellm import token_counter  # type: ignore

        return int(token_counter(model=model_name, text=content) or 0)
    except Exception:
        return max(1, len(content) // 4)


def install_engine_usage_tracker(engine: object, model_name: str):
    """
    Monkey-patch engine.generate() to collect per-job token/cost/time stats.
    """
    lock = threading.Lock()
    stats: Dict[str, Dict[str, float]] = {}

    pricing_key = normalize_model_for_pricing(model_name)
    pricing = MODEL_PRICING.get(pricing_key, {"input": 0.15, "output": 0.60})

    original_generate = engine.generate

    def wrapped_generate(content, system_prompt=None, **kwargs):
        job_key = _CURRENT_JOB.get()
        start = time.perf_counter()
        resp = original_generate(content, system_prompt=system_prompt, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if isinstance(content, list):
            prompt_text = "\n".join(str(x) for x in content)
        else:
            prompt_text = str(content or "")
        if system_prompt:
            prompt_text = f"{system_prompt}\n{prompt_text}"

        completion_text = str(resp or "")
        prompt_tokens = estimate_tokens(pricing_key, prompt_text)
        completion_tokens = estimate_tokens(pricing_key, completion_text)
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = (
            (prompt_tokens / 1_000_000.0) * pricing["input"]
            + (completion_tokens / 1_000_000.0) * pricing["output"]
        )

        with lock:
            bucket = stats.setdefault(
                job_key,
                {
                    "request_count": 0.0,
                    "prompt_tokens": 0.0,
                    "completion_tokens": 0.0,
                    "total_tokens": 0.0,
                    "estimated_cost_usd": 0.0,
                    "llm_elapsed_ms": 0.0,
                },
            )
            bucket["request_count"] += 1
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens
            bucket["total_tokens"] += total_tokens
            bucket["estimated_cost_usd"] += estimated_cost
            bucket["llm_elapsed_ms"] += elapsed_ms

        return resp

    engine.generate = wrapped_generate
    return stats, lock


def build_evaluation_instruction() -> str:
    """
    TextGrad loss system instruction (must be English).
    The model output will be used as feedback to compute the textual gradient.
    """
    return (
        "You are an expert evaluator for classroom teaching transcripts. "
        "The transcript you receive is a spoken teaching script for one 6-minute window. "
        "Your feedback will be used by a text-optimization algorithm to rewrite the transcript. "
        "Therefore, you MUST NOT rewrite the transcript yourself. "
        "Instead, provide a short, critical ENGLISH critique describing concrete edits to improve: "
        "(1) clarity and structure, (2) pacing, (3) transitions/signposting, and "
        "(4) opportunities to engage learners (e.g., questions, participation prompts, checks for understanding). "
        "Be strict and specific: point out where the transcript is unclear, overly long, repetitive, "
        "missing transitions, or missing engagement cues. "
        "CRUCIAL: Do not invent any new facts, events, dates, names, or numeric details. "
        "When suggesting edits, instruct changes that preserve the original teaching meaning and factual content. "
        "Output ONLY English feedback (no JSON, no headings)."
    )


def _has_detectable_student_speech(text: str) -> bool:
    t = text or ""
    cues = [
        r"\bstudent\s*[:：]",
        r"\bstudents\s*[:：]",
        r"\bs\d+\s*[:：]",
        r"student\s*[:：]",
        r"\blearner\s*[:：]",
    ]
    return any(re.search(p, t, flags=re.IGNORECASE) for p in cues)


def predict_expected_icap_level(engine: object, model_name: str, transcript_text: str) -> Tuple[str, str]:
    """
    Predict expected ICAP level as an ordinal classification (p/a < c < i).
    Minimal fallback: if no student speech detected, force p/a.
    """
    if not _has_detectable_student_speech(transcript_text):
        return "p/a", "No detectable student speech; forced to p/a."

    prompt = f"""
You are an ICAP classifier.
This is an ORDINAL prediction task: p/a < c < i.

Definitions:
- p/a: no student speech, or only passive/attentive behavior without constructive output.
- c: individual constructive output (explain, justify, generate), but no explicit peer-to-peer co-construction.
- i: explicit peer interaction/co-construction (discussion/debate/peer critique/collaborative reasoning).

Output strict JSON:
{{
  "expected_icap_level": "p/a|c|i",
  "reason": "short evidence-based reason"
}}

[Transcript]
{transcript_text[:MAX_CHARS]}
""".strip()

    try:
        raw = engine.generate(
            content=prompt,
            system_prompt="Return JSON only. Follow ordinal ICAP definitions strictly.",
        )
        text = str(raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```json\\s*|```$", "", text, flags=re.MULTILINE).strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        obj = json.loads(text)
        lvl = str(obj.get("expected_icap_level", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()
        if lvl in {"p/a", "c", "i"}:
            return lvl, reason or "Predicted by ordinal ICAP classifier."
    except Exception:
        pass

    # Heuristic fallback (minimal)
    lower = (transcript_text or "").lower()
    interactive_cues = [
        "discuss", "debate", "pair", "partner", "group", "team",
        "peer", "collaborat", "each other", "with your partner", "in groups",
    ]
    if any(c in lower for c in interactive_cues):
        return "i", "Fallback heuristic: detected peer-interaction cues."
    return "c", "Fallback heuristic: student speech detected without explicit peer interaction."


def build_evaluation_instruction_for_target(target_icap_level: str, target_icap_reason: str) -> str:
    level = (target_icap_level or "c").strip().lower()
    mode_hint = (
        "Target ICAP is i: encourage explicit peer interaction, dialogue, and co-construction."
        if level == "i"
        else "Target ICAP is c: encourage individual constructive reasoning without requiring peer interaction."
        if level == "c"
        else "Target ICAP is p/a: keep conservative structure; do not fabricate student speech."
    )
    return (
        "You are an expert evaluator for classroom teaching transcripts. "
        "The transcript you receive is a spoken teaching script for one 6-minute window. "
        "Your feedback will be used by a text-optimization algorithm to rewrite the transcript. "
        "Therefore, you MUST NOT rewrite the transcript yourself. "
        "Instead, provide a short, critical ENGLISH critique describing concrete edits to improve: "
        "(1) clarity and structure, (2) pacing, (3) transitions/signposting, and "
        "(4) opportunities to engage learners (e.g., questions, participation prompts, checks for understanding). "
        "Be strict and specific: point out where the transcript is unclear, overly long, repetitive, "
        "missing transitions, or missing engagement cues. "
        "CRUCIAL: Do not invent any new facts, events, dates, names, or numeric details. "
        "When suggesting edits, instruct changes that preserve the original teaching meaning and factual content. "
        f"ICAP ordinal target for this sample: {level} (reason: {target_icap_reason}). "
        f"{mode_hint} "
        "Output ONLY English feedback (no JSON, no headings)."
    )


def main():
    run_start = time.perf_counter()
    transcripts = list_segments_from_json(INPUT_JSON_PATH)
    if not transcripts:
        raise RuntimeError(f"No valid segments[].text found in: {INPUT_JSON_PATH}")

    model_name, engine = resolve_engine_and_model()
    usage_stats, usage_lock = install_engine_usage_tracker(engine, model_name)
    tg.set_backward_engine(engine, override=True)

    # BlackboxLLM is not strictly required here, but keeping it ensures
    # TextGrad can always make an LLM call if needed by internal components.
    _model = BlackboxLLM(engine)

    output_parent = Path(OUTPUT_JSONL_PATH).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    constraints = [
        "Preserve the original factual meaning: do not invent new events, names, dates, deadlines, course requirements, or numeric details.",
        "Do not change the core teaching intent; rewrite only for clarity, structure, pacing, and engagement cues.",
        "Maintain the original topic sequence and keep a coherent timeline within the 6-minute window.",
        "Keep the instructor's tone and approximate speaking style; avoid adding unrelated content.",
    ]

    def process_one(item: Tuple[str, str, str]) -> Dict:
        window_label, transcript_path, transcript_text = item
        job_start = time.perf_counter()
        token = _CURRENT_JOB.set(window_label)
        try:
            if not transcript_text:
                return {
                    "window": window_label,
                    "input_transcript_path": transcript_path,
                    "model_name": model_name,
                    "steps": STEPS,
                    "step_records": [],
                    "optimized_transcript": "",
                    "error": "empty transcript",
                    "efficiency": {
                        "elapsed_ms": 0,
                        "request_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "llm_elapsed_ms": 0,
                    },
                }

            transcript_for_opt = transcript_text[:MAX_CHARS] if MAX_CHARS > 0 else transcript_text
            expected_icap_level, expected_icap_reason = predict_expected_icap_level(
                engine, model_name, transcript_for_opt
            )
            evaluation_instruction = build_evaluation_instruction_for_target(
                expected_icap_level, expected_icap_reason
            )
            transcript_var = Variable(
                transcript_for_opt,
                requires_grad=True,
                role_description="classroom teaching transcript",
            )
            loss_fn = tg.TextLoss(evaluation_instruction)
            optimizer = tg.TGD(parameters=[transcript_var], constraints=constraints)

            step_records: List[Dict] = []
            for step in range(STEPS):
                loss = loss_fn(transcript_var)
                loss_text = str(loss.value)
                transcript_var.reset_gradients()
                loss.backward()
                text_gradient = transcript_var.get_gradient_text()
                optimizer.step()
                step_records.append(
                    {"step": step, "loss_text": loss_text, "text_gradient": text_gradient}
                )

            optimized_transcript = transcript_var.value
            with usage_lock:
                s = usage_stats.get(window_label, {}).copy()
            return {
                "window": window_label,
                "input_transcript_path": transcript_path,
                "model_name": model_name,
                "expected_icap_level": expected_icap_level,
                "expected_icap_reason": expected_icap_reason,
                "steps": STEPS,
                "step_records": step_records,
                "optimized_transcript": optimized_transcript,
                "efficiency": {
                    "elapsed_ms": int((time.perf_counter() - job_start) * 1000),
                    "request_count": int(s.get("request_count", 0)),
                    "prompt_tokens": int(s.get("prompt_tokens", 0)),
                    "completion_tokens": int(s.get("completion_tokens", 0)),
                    "total_tokens": int(s.get("total_tokens", 0)),
                    "estimated_cost_usd": round(float(s.get("estimated_cost_usd", 0.0)), 6),
                    "llm_elapsed_ms": int(s.get("llm_elapsed_ms", 0)),
                },
            }
        finally:
            _CURRENT_JOB.reset(token)

    print(f"[INFO] samples={len(transcripts)} workers={MAX_WORKERS}")
    indexed = list(enumerate(transcripts))
    results: List[Dict] = [None] * len(indexed)  # type: ignore
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, item): idx for idx, item in indexed}
        for f in as_completed(futures):
            idx = futures[f]
            item = indexed[idx][1]
            try:
                r = f.result()
                results[idx] = r
                print(f"[DONE] {r['window']} tokens={r['efficiency']['total_tokens']} cost=${r['efficiency']['estimated_cost_usd']}")
            except Exception as e:
                window_label, transcript_path, _ = item
                results[idx] = {
                    "window": window_label,
                    "input_transcript_path": transcript_path,
                    "model_name": model_name,
                    "steps": STEPS,
                    "step_records": [],
                    "optimized_transcript": "",
                    "error": str(e),
                    "efficiency": {
                        "elapsed_ms": 0,
                        "request_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "llm_elapsed_ms": 0,
                    },
                }
                print(f"[FAIL] {window_label}: {e}")

    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as out_f:
        for out_item in results:
            out_f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    total_prompt_tokens = sum(int(i.get("efficiency", {}).get("prompt_tokens", 0)) for i in results)
    total_completion_tokens = sum(int(i.get("efficiency", {}).get("completion_tokens", 0)) for i in results)
    total_tokens = sum(int(i.get("efficiency", {}).get("total_tokens", 0)) for i in results)
    total_cost = round(sum(float(i.get("efficiency", {}).get("estimated_cost_usd", 0.0)) for i in results), 6)
    total_llm_elapsed_ms = sum(int(i.get("efficiency", {}).get("llm_elapsed_ms", 0)) for i in results)
    elapsed_sec = round(time.perf_counter() - run_start, 3)

    pricing_model = normalize_model_for_pricing(model_name)
    pricing = MODEL_PRICING.get(pricing_model, {"input": 0.15, "output": 0.60})
    summary = {
        "input_json_path": INPUT_JSON_PATH,
        "output_jsonl_path": OUTPUT_JSONL_PATH,
        "model_name": model_name,
        "pricing_model_reference": pricing_model,
        "pricing_per_1m_tokens": {
            "input": pricing["input"],
            "output": pricing["output"],
        },
        "workers": MAX_WORKERS,
        "samples": len(results),
        "elapsed_sec": elapsed_sec,
        "llm_elapsed_sec_sum": round(total_llm_elapsed_ms / 1000.0, 3),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "estimated_total_cost_usd": total_cost,
    }
    summary_path = str(Path(OUTPUT_JSONL_PATH).with_suffix(".summary.json"))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n[SAVED] {OUTPUT_JSONL_PATH}")
    print(f"[SAVED] {summary_path}")
    print(
        f"[USAGE] time={elapsed_sec}s, tokens={total_tokens} "
        f"(prompt={total_prompt_tokens}, completion={total_completion_tokens}), cost=${total_cost}"
    )


if __name__ == "__main__":
    main()

