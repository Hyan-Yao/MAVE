"""
Teaching transcript optimization + Reflexion components from the original hotpotqa_runs.

Directly reuses assets from reflexion/hotpotqa_runs:
  - prompts.reflect_prompt (reflection prompt template from the paper)
  - fewshots.REFLECTIONS (few-shot reflection examples)
  - prompts.REFLECTION_HEADER / LAST_TRIAL_HEADER (injection format aligned with ReactReflectAgent)

This script does not directly call ReactReflectAgent / ReAct because those classes
strongly depend on Wikipedia, QA questions, and action parsing. Transcript optimization
is treated as a standalone task while preserving the Reflexion protocol:
"generate reflection + inject into next iteration".

"""
import os
import re
import sys
import json
import time
import enum
import threading
import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from openai import OpenAI

# ---------- LLM CONFIG ----------
# Consistent with project_class/focus_scoring_fine.py: prefer OpenRouter, fallback to official OpenAI
TIMEOUT = 60

_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_openai_key = os.getenv("OPENAI_API_KEY")

if _openrouter_key and _openrouter_key.strip():
    # Use OpenRouter
    MODEL = "openai/gpt-4o-mini"
    client = OpenAI(
        api_key=_openrouter_key.strip(),
        base_url="https://openrouter.ai/api/v1",
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


def _api_provider_label() -> str:
    if _openrouter_key and _openrouter_key.strip():
        return "openrouter"
    return "openai"


# --- Aligned with original reflexion/hotpotqa_runs ---
_HOTPOT_ROOT = os.getenv(
    "REFLEXION_HOTPOT_ROOT",
    "",
)
if _HOTPOT_ROOT not in sys.path:
    sys.path.insert(0, _HOTPOT_ROOT)

# fewshots has no third-party dependency; prompts depends on langchain.
# If unavailable, fallback to equivalent plain-string templates (consistent with prompts.py).
try:
    from fewshots import REFLECTIONS  # type: ignore  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"Failed to import hotpotqa_runs.fewshots (does path exist: {_HOTPOT_ROOT}?)"
    ) from e

_REFLECT_PROMPT_SOURCE = "embedded_fallback_matching_prompts_py"

try:
    from prompts import reflect_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER  # type: ignore  # noqa: E402

    _REFLECT_PROMPT_SOURCE = "reflexion.hotpotqa_runs.prompts"

    def build_reflect_user_prompt(
        examples: str, question: str, scratchpad: str
    ) -> str:
        return reflect_prompt.format(
            examples=examples, question=question, scratchpad=scratchpad
        )

except ImportError:
    # Keep wording consistent with REFLECT_INSTRUCTION / HEADER from prompts.py (avoid langchain dependency)
    REFLECTION_HEADER = (
        "You have attempted to answer following question before and failed. "
        "The following reflection(s) give a plan to avoid failing to answer the question in "
        "the same way you did previously. Use them to improve your strategy of correctly "
        "answering the given question.\n"
    )
    LAST_TRIAL_HEADER = (
        "You have attempted to answer the following question before and failed. "
        "Below is the last trial you attempted to answer the question.\n"
    )
    _REFLECT_INSTRUCTION_FALLBACK = (
        "You are an advanced reasoning agent that can improve based on self refection. You will be given "
        "a previous reasoning trial in which you were given access to an Docstore API environment and "
        "a question to answer. You were unsuccessful in answering the question either because you guessed "
        "the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a "
        "few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan "
        "that aims to mitigate the same failure. Use complete sentences.  \n"
        "Here are some examples:\n"
        "{examples}\n"
        "\n"
        "Previous trial:\n"
        "Question: {question}{scratchpad}\n"
        "\n"
        "Reflection:"
    )

    def build_reflect_user_prompt(
        examples: str, question: str, scratchpad: str
    ) -> str:
        return _REFLECT_INSTRUCTION_FALLBACK.format(
            examples=examples, question=question, scratchpad=scratchpad
        )


KNOWLEDGE_SEGMENTS_JSON = os.getenv(
    "REFLEXION_KNOWLEDGE_SEGMENTS_JSON",
    "",
)
OUTPUT_DIR = os.getenv(
    "REFLEXION_OUTPUT_DIR",
    "",
)
OUTPUT_JSONL_PATH = os.getenv(
    "REFLEXION_REPO_OUTPUT_JSONL",
    os.path.join(OUTPUT_DIR, "reflexion_transcript.jsonl"),
)
OUTPUT_REFLECTION_JSONL_PATH = os.getenv(
    "REFLEXION_REPO_SIGNAL_JSONL",
    os.path.join(OUTPUT_DIR, "reflection_signals.jsonl"),
)

MAX_ITERS = int(os.getenv("REFLEXION_MAX_ITERS", "5"))
MAX_CHARS = int(os.getenv("REFLEXION_MAX_CHARS", "12000"))
REFLECT_SCRATCHPAD_MAX = int(os.getenv("REFLEXION_REFLECT_SCRATCHPAD_MAX", "8000"))
TEMPERATURE = float(os.getenv("REFLEXION_TEMPERATURE", "0.2"))
RETRY_TIMES = int(os.getenv("REFLEXION_RETRY_TIMES", "5"))
MAX_WORKERS = max(1, int(os.getenv("REFLEXION_MAX_WORKERS", "20")))
PROMPT_COST_PER_1M = float(os.getenv("REFLEXION_PROMPT_COST_PER_1M", "0.15"))
COMPLETION_COST_PER_1M = float(os.getenv("REFLEXION_COMPLETION_COST_PER_1M", "0.60"))

# reflexion: accumulate verbal reflections
# (aligned with ReactReflectAgent + ReflexionStrategy.REFLEXION)
# last_trial: inject only the last trial trace
# (aligned with ReflexionStrategy.LAST_ATTEMPT, without separate verbal reflection)
_REF_STRAT = (os.getenv("REFLEXION_STRATEGY") or "reflexion").strip().lower()


class TranscriptReflexionStrategy(str, enum.Enum):
    REFLEXION = "reflexion"
    LAST_TRIAL = "last_trial"


TRANSCRIPT_TASK_QUESTION = (
    "Improve this classroom teaching transcript for clarity, pacing, and student engagement "
    "while preserving factual content. The transcript may be Chinese or English; "
    "if rewriting, prefer clear, classroom-ready English unless the source is already another target language."
)

# If few-shot examples are too long, disable with: set REFLEXION_USE_HOTPOT_FEWSHOT=0
USE_HOTPOT_FEWSHOT = (os.getenv("REFLEXION_USE_HOTPOT_FEWSHOT") or "1").strip() not in (
    "0",
    "false",
    "False",
)

CURRENT_TASK_KEY = contextvars.ContextVar("reflexion_current_task_key", default="global")
USAGE_LOCK = threading.Lock()
USAGE_TOTALS = {
    "request_count": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
    "llm_elapsed_ms": 0.0,
}
USAGE_BY_TASK: Dict[str, Dict[str, float]] = {}


def _has_detectable_student_speech(transcript_text: str) -> bool:
    text = transcript_text or ""
    cues = [
        r"\bstudent\s*[:：]",
        r"\bstudents\s*[:：]",
        r"\bs\d+\s*[:：]",
        r"student\s*[:：]",
        r"\blearner\s*[:：]",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in cues)


def infer_expected_icap_level(
    client: OpenAI, model: str, transcript_text: str
) -> Tuple[str, str]:
    """
    Infer expected ICAP level with ordinal prediction guidance.
    Hard rule from user:
    - if no detectable student speech -> p/a
    """
    if not _has_detectable_student_speech(transcript_text):
        return "p/a", "No detectable student speech; force label to p/a."

    system_prompt = (
        "You are an ICAP classifier. This is an ordinal prediction task with ordered labels: p/a < c < i. "
        "Return JSON only."
    )
    user_prompt = f"""
Classify the transcript into ONE expected ICAP label using definitions:
- p/a: no student speech, or only passive/attentive behavior without constructive output.
- c: students produce individual constructive outputs (explain, justify, generate) but no explicit peer-to-peer co-construction.
- i: explicit peer interaction/co-construction (discussion, debate, critique, collaborative reasoning among students).

Important:
- This is ORDINAL prediction, not nominal.
- Output exactly one label from: "p/a", "c", "i".
- If evidence is ambiguous between adjacent levels, pick the lower one.

Output JSON schema:
{{
  "expected_icap_level": "p/a|c|i",
  "reason": "short evidence-based reason"
}}

[Transcript]
{truncate_block(transcript_text, MAX_CHARS)}
""".strip()

    raw = chat_once(client, model, system_prompt, user_prompt).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        obj = json.loads(raw)
        lvl = str(obj.get("expected_icap_level", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()
        if lvl in {"p/a", "c", "i"}:
            return lvl, reason or "LLM ICAP ordinal prediction."
    except Exception:
        pass

    # minimal fallback
    txt = (transcript_text or "").lower()
    interactive_cues = [
        "discuss", "debate", "pair", "partner", "group", "team",
        "peer", "collaborat", "each other", "with your partner", "in groups",
    ]
    if any(cue in txt for cue in interactive_cues):
        return "i", "Fallback heuristic: detected peer-interaction cues."
    return "c", "Fallback heuristic: detected student speech without explicit peer interaction."


def load_knowledge_segments(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Knowledge segments JSON does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("Invalid knowledge segments JSON: `segments` must be a list.")
    cleaned: List[Dict] = []
    for idx, seg in enumerate(segments, start=1):
        if not isinstance(seg, dict):
            continue
        kp_id = seg.get("knowledge_point_id", idx)
        try:
            kp_id = int(kp_id)
        except Exception:
            kp_id = idx
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        cleaned.append(
            {
                "knowledge_point_id": kp_id,
                "topic": str(seg.get("topic", "")).strip(),
                "summary": str(seg.get("summary", "")).strip(),
                "start_char": seg.get("start_char"),
                "end_char": seg.get("end_char"),
                "text": text,
            }
        )
    cleaned.sort(key=lambda x: x["knowledge_point_id"])
    return cleaned


def chat_once(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    last_error = None
    for attempt in range(RETRY_TIMES):
        try:
            call_start = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                timeout=TIMEOUT,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            elapsed_ms = (time.perf_counter() - call_start) * 1000.0
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
            if total_tokens <= 0:
                total_tokens = prompt_tokens + completion_tokens
            estimated_cost = (
                (prompt_tokens / 1_000_000.0) * PROMPT_COST_PER_1M
                + (completion_tokens / 1_000_000.0) * COMPLETION_COST_PER_1M
            )
            task_key = CURRENT_TASK_KEY.get()
            with USAGE_LOCK:
                USAGE_TOTALS["request_count"] += 1
                USAGE_TOTALS["prompt_tokens"] += prompt_tokens
                USAGE_TOTALS["completion_tokens"] += completion_tokens
                USAGE_TOTALS["total_tokens"] += total_tokens
                USAGE_TOTALS["estimated_cost_usd"] += estimated_cost
                USAGE_TOTALS["llm_elapsed_ms"] += elapsed_ms
                bucket = USAGE_BY_TASK.setdefault(
                    task_key,
                    {
                        "request_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
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
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            if attempt < RETRY_TIMES - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {RETRY_TIMES} retries: {last_error}")


# --- String helpers aligned with agents.py (avoid importing agents and pulling Wikipedia / langchain.docstore) ---
def format_step(step: str) -> str:
    return step.strip("\n").strip().replace("\n", "")


def format_reflections(reflections: List[str]) -> str:
    if not reflections:
        return ""
    return (
        REFLECTION_HEADER
        + "Reflections:\n- "
        + "\n- ".join([r.strip() for r in reflections])
    )


def format_last_attempt(question: str, scratchpad: str) -> str:
    return (
        LAST_TRIAL_HEADER
        + f"Question: {question}\n"
        + scratchpad.strip()
        + "\n(END PREVIOUS TRIAL)\n"
    )


def truncate_block(text: str, limit: int) -> str:
    t = text.strip()
    if len(t) <= limit:
        return t
    head = limit // 2
    tail = limit - head
    return t[:head] + "\n...[truncated for max token safety]...\n" + t[-tail:]


def build_hotpot_style_scratchpad(feedback: str, candidate: str) -> str:
    """Pack evaluator feedback + current draft into a trial record consumable by reflect_prompt."""
    fb = truncate_block(feedback, REFLECT_SCRATCHPAD_MAX // 3)
    cand = truncate_block(candidate, 2 * REFLECT_SCRATCHPAD_MAX // 3)
    return (
        "\nThought 1: I should review evaluator feedback on my current transcript draft.\n"
        "Action 1: Submit[draft]\n"
        f"Observation 1:\n{fb}\n\n"
        "Current draft transcript:\n"
        f"{cand}\n"
    )


def reflection_fewshot_block() -> str:
    if USE_HOTPOT_FEWSHOT:
        return REFLECTIONS
    return "(Task adapted from HotPotQA-style Reflexion; few-shot examples omitted to save tokens.)\n"


def run_hotpot_reflect_llm(
    client: OpenAI,
    model: str,
    feedback: str,
    candidate: str,
) -> str:
    """Use reflect_prompt + REFLECTIONS from the repo to generate verbal reflection."""
    scratchpad = truncate_block(
        build_hotpot_style_scratchpad(feedback, candidate),
        REFLECT_SCRATCHPAD_MAX,
    )
    user_prompt = build_reflect_user_prompt(
        examples=reflection_fewshot_block(),
        question=TRANSCRIPT_TASK_QUESTION,
        scratchpad=scratchpad,
    )
    system_prompt = (
        "Follow the user instructions exactly. Output only the text that belongs after 'Reflection:' "
        "(a short diagnosis plus a concise new plan), without repeating the examples block."
    )
    return format_step(chat_once(client, model, system_prompt, user_prompt))


def evaluator_feedback(
    client: OpenAI,
    model: str,
    original: str,
    candidate: str,
) -> str:
    user = f"""
Evaluate the candidate teaching transcript against the original for teaching quality.

Return in English with sections: Strengths / Weaknesses / Top 3 priorities.
Do not rewrite the transcript.

[Original]
{truncate_block(original, MAX_CHARS)}

[Candidate]
{truncate_block(candidate, MAX_CHARS)}
""".strip()
    return chat_once(
        client,
        model,
        "You are a strict instructional coach.",
        user,
    ).strip()


def revise_with_reflexion_memory(
    client: OpenAI,
    model: str,
    original: str,
    candidate: str,
    reflections_section: str,
    target_icap_level: str,
    target_icap_reason: str,
) -> str:
    user = f"""
{reflections_section}

Task question:
{TRANSCRIPT_TASK_QUESTION}

[ICAP ordinal guidance for this revision]
- This is an ordinal target: p/a < c < i.
- Current expected ICAP target for this knowledge point: {target_icap_level}
- Why: {target_icap_reason}
- Revision requirement: keep factual content unchanged, but optimize pedagogical expression so the revised transcript is better aligned with target ICAP level.
- If target is c: strengthen individual constructive expression.
- If target is i: strengthen explicit peer-to-peer interaction cues and co-construction.
- If target is p/a: do NOT fabricate student speech.

[Original transcript — factual grounding]
{truncate_block(original, MAX_CHARS)}

[Current draft to revise]
{truncate_block(candidate, MAX_CHARS)}

Output ONLY the full improved transcript text.
""".strip()
    return chat_once(
        client,
        model,
        "You are a precise teaching-transcript editor.",
        user,
    ).strip()


def run_single_transcript(
    client: OpenAI,
    model: str,
    transcript_text: str,
    strategy: TranscriptReflexionStrategy,
) -> Dict:
    original = transcript_text[:MAX_CHARS].strip()
    candidate = original
    step_records: List[Dict] = []
    verbal_reflections: List[str] = []
    target_icap_level, target_icap_reason = infer_expected_icap_level(client, model, original)

    for step in range(MAX_ITERS):
        fb = evaluator_feedback(client, model, original, candidate)
        injected: str
        verbal: str

        if strategy == TranscriptReflexionStrategy.LAST_TRIAL:
            trial_pad = truncate_block(
                build_hotpot_style_scratchpad(fb, candidate),
                REFLECT_SCRATCHPAD_MAX,
            )
            injected = format_last_attempt(TRANSCRIPT_TASK_QUESTION, trial_pad)
            verbal = trial_pad
        else:
            verbal = run_hotpot_reflect_llm(client, model, fb, candidate)
            verbal_reflections.append(verbal)
            injected = format_reflections(verbal_reflections)

        improved = revise_with_reflexion_memory(
            client,
            model,
            original,
            candidate,
            injected,
            target_icap_level,
            target_icap_reason,
        ).strip()
        if not improved:
            improved = candidate

        step_records.append(
            {
                "step": step,
                "strategy": strategy.value,
                "evaluator_feedback": fb,
                "reflection_signal": verbal,
                "reflexion_injected_block": injected,
                "target_icap_level_for_revision": target_icap_level,
                "target_icap_reason_for_revision": target_icap_reason,
                "candidate_before": candidate,
                "candidate_after": improved,
                "accumulated_verbal_reflections": list(verbal_reflections),
            }
        )
        candidate = improved

    return {
        "reflexion_strategy": strategy.value,
        "max_iters": MAX_ITERS,
        "used_hotpot_fewshot": USE_HOTPOT_FEWSHOT,
        "target_icap_level_for_generation": target_icap_level,
        "target_icap_reason_for_generation": target_icap_reason,
        "original_transcript": original,
        "optimized_transcript": candidate,
        "step_records": step_records,
    }


def run_single_knowledge_point(
    seg: Dict,
    strategy: TranscriptReflexionStrategy,
    provider: str,
) -> Tuple[Dict, List[Dict]]:
    task_start = time.perf_counter()
    kp_id = int(seg["knowledge_point_id"])
    label = f"kp_{kp_id}"
    text = str(seg["text"]).strip()
    if not text:
        raise ValueError(f"knowledge_point_id={kp_id} has empty text.")
    token = CURRENT_TASK_KEY.set(label)
    try:
        result = run_single_transcript(client, MODEL, text, strategy)
    finally:
        CURRENT_TASK_KEY.reset(token)

    with USAGE_LOCK:
        local_usage = dict(USAGE_BY_TASK.get(label, {}))
    local_usage.setdefault("request_count", 0)
    local_usage.setdefault("prompt_tokens", 0)
    local_usage.setdefault("completion_tokens", 0)
    local_usage.setdefault("total_tokens", 0)
    local_usage.setdefault("estimated_cost_usd", 0.0)
    local_usage.setdefault("llm_elapsed_ms", 0.0)
    local_usage["elapsed_sec"] = round(time.perf_counter() - task_start, 3)
    local_usage["estimated_cost_usd"] = round(float(local_usage["estimated_cost_usd"]), 6)
    local_usage["llm_elapsed_ms"] = round(float(local_usage["llm_elapsed_ms"]), 3)
    expected_icap_level, expected_icap_reason = infer_expected_icap_level(
        client,
        MODEL,
        str(result.get("optimized_transcript", "")),
    )

    item = {
        "knowledge_point_id": kp_id,
        "topic": seg.get("topic", ""),
        "summary": seg.get("summary", ""),
        "source_knowledge_segments_json": KNOWLEDGE_SEGMENTS_JSON,
        "provider": provider,
        "model": MODEL,
        "reflect_prompt_resolved_from": _REFLECT_PROMPT_SOURCE,
        "hotpotqa_fewshots": "reflexion.hotpotqa_runs.fewshots.REFLECTIONS"
        if USE_HOTPOT_FEWSHOT
        else "(omitted)",
        "expected_icap_level": expected_icap_level,
        "expected_icap_reason": expected_icap_reason,
        "efficiency": local_usage,
        **result,
    }
    signals = []
    for rec in result["step_records"]:
        signals.append(
            {
                "knowledge_point_id": kp_id,
                "topic": seg.get("topic", ""),
                "step": rec["step"],
                "strategy": rec["strategy"],
                "reflection_signal": rec["reflection_signal"],
            }
        )
    return item, signals


def main() -> None:
    run_start = time.perf_counter()
    if not os.path.isdir(_HOTPOT_ROOT):
        raise FileNotFoundError(
            f"hotpotqa_runs directory not found, please check: {_HOTPOT_ROOT}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        strategy = TranscriptReflexionStrategy(_REF_STRAT)
    except ValueError:
        raise ValueError(
            f"Invalid REFLEXION_STRATEGY={_REF_STRAT!r}; use reflexion or last_trial"
        )

    segments = load_knowledge_segments(KNOWLEDGE_SEGMENTS_JSON)
    if not segments:
        raise ValueError(f"No usable segments found in: {KNOWLEDGE_SEGMENTS_JSON}")

    provider = _api_provider_label()
    summary_path = os.getenv(
        "REFLEXION_REPO_SUMMARY_JSON",
        os.path.join(OUTPUT_DIR, "reflexion_run_summary.json"),
    )
    print(
        f"[START] strategy={strategy.value} model={MODEL} provider={provider} "
        f"knowledge_points={len(segments)} workers={MAX_WORKERS}"
    )

    ordered_items: List[Dict] = [None] * len(segments)  # type: ignore
    ordered_signals: List[List[Dict]] = [None] * len(segments)  # type: ignore
    lock = threading.Lock()
    done_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(run_single_knowledge_point, seg, strategy, provider): idx
            for idx, seg in enumerate(segments)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            kp_id = segments[idx]["knowledge_point_id"]
            try:
                item, signals = fut.result()
                ordered_items[idx] = item
                ordered_signals[idx] = signals
                with lock:
                    done_count += 1
                    print(
                        f"[DONE] kp_{kp_id} steps={len(item.get('step_records', []))} "
                        f"({done_count}/{len(segments)})"
                    )
            except Exception as e:
                with lock:
                    done_count += 1
                    print(f"[FAIL] kp_{kp_id} ({done_count}/{len(segments)}): {e}")

    valid_items = [x for x in ordered_items if isinstance(x, dict)]
    valid_signals = [s for s in ordered_signals if isinstance(s, list)]

    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as out_all:
        for item in valid_items:
            out_all.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(OUTPUT_REFLECTION_JSONL_PATH, "w", encoding="utf-8") as out_sig:
        for signal_list in valid_signals:
            for rec in signal_list:
                out_sig.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with USAGE_LOCK:
        totals = dict(USAGE_TOTALS)
    elapsed_sec = round(time.perf_counter() - run_start, 3)
    totals["estimated_cost_usd"] = round(float(totals.get("estimated_cost_usd", 0.0)), 6)
    totals["llm_elapsed_ms"] = round(float(totals.get("llm_elapsed_ms", 0.0)), 3)
    throughput_kp_per_min = round((len(valid_items) / max(1e-9, elapsed_sec)) * 60.0, 3)
    summary = {
        "source_knowledge_segments_json": KNOWLEDGE_SEGMENTS_JSON,
        "provider": provider,
        "model": MODEL,
        "strategy": strategy.value,
        "knowledge_points_total": len(segments),
        "knowledge_points_succeeded": len(valid_items),
        "max_workers": MAX_WORKERS,
        "elapsed_sec": elapsed_sec,
        "throughput_kp_per_min": throughput_kp_per_min,
        "pricing": {
            "prompt_cost_per_1m": PROMPT_COST_PER_1M,
            "completion_cost_per_1m": COMPLETION_COST_PER_1M,
        },
        "usage_totals": totals,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print(
        f"\nDone.\nKnowledge source: {KNOWLEDGE_SEGMENTS_JSON}\n"
        f"Full output: {OUTPUT_JSONL_PATH}\nReflection signals: {OUTPUT_REFLECTION_JSONL_PATH}\n"
        f"Summary: {summary_path}"
    )
    print(
        f"[USAGE] requests={totals.get('request_count', 0)}, "
        f"tokens={totals.get('total_tokens', 0)} "
        f"(prompt={totals.get('prompt_tokens', 0)}, completion={totals.get('completion_tokens', 0)}), "
        f"elapsed={elapsed_sec}s, throughput={throughput_kp_per_min} kp/min, "
        f"cost≈${totals.get('estimated_cost_usd', 0.0)}"
    )


if __name__ == "__main__":
    main()
