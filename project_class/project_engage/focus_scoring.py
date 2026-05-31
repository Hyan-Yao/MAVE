'''
Engagement scoring (fine): read minute-level and grid-level preprocessor output, score every 3-minute window.
LLM → anchor + acceptable_deviation → reconstruct_confidence → expected MAE → engagement.

调参重点：alpha，alpha越小，confidence越偏向于anchor，alpha越大，confidence越偏向于0和3。
[USAGE] requests=447, total_prompt_tokens=538004, total_completion_tokens=54132, avg_prompt_tokens_per_req=1203.6, avg_completion_tokens_per_req=121.1, elapsed_sec=193.44, req_per_sec=2.311, tokens_per_sec=3061.1
[COST] prompt=$0.0807, completion=$0.0325, total=$0.1132
[SAVED] /Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/HIS 101 (4) - ZOOM Class from Fri. Aug 21st - Syllabus Questions and Announcements for next week/focus_output_now/focus_scoring_efficiency_report.json

[USAGE] requests=216, total_prompt_tokens=254068, total_completion_tokens=25807, avg_prompt_tokens_per_req=1176.2, avg_completion_tokens_per_req=119.5, elapsed_sec=87.30, req_per_sec=2.474, tokens_per_sec=3206.0
[COST] prompt=$0.0381, completion=$0.0155, total=$0.0536
[SAVED] /Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Town Hall Video for MSA 601 January 12, 2026/focus_output_now/focus_scoring_efficiency_report.json

[USAGE] requests=241, total_prompt_tokens=285349, total_completion_tokens=29778, avg_prompt_tokens_per_req=1184.0, avg_completion_tokens_per_req=123.6, elapsed_sec=81.07, req_per_sec=2.973, tokens_per_sec=3887.0
[COST] prompt=$0.0428, completion=$0.0179, total=$0.0607
[SAVED] /Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/HIS 101 (4) - ZOOM Class from Fri. Aug 21st - Syllabus Questions and Announcements for next week/focus_output_now/focus_scoring_efficiency_report.json
'''

# client = OpenAI(
#     api_key=os.environ["OPENAI_API_KEY"]
# )
# MODEL = "gpt-4o-mini"
# TIMEOUT = 60

import os
import json
from openai import OpenAI
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- LLM CONFIG ----------
# Supports two modes: prefer OpenRouter; fallback to official OpenAI API
TIMEOUT = 60
MAX_WORKERS = 10
# Cost model (USD per 1M tokens). Update if provider pricing changes.
PROMPT_COST_PER_1M = 0.15
COMPLETION_COST_PER_1M = 0.60

TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0
USAGE_LOCK = threading.Lock()

_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_openai_key = os.getenv("OPENAI_API_KEY")

if _openrouter_key and _openrouter_key.strip():
    # Use OpenRouter
    MODEL = "openai/gpt-4o-mini"
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

# Input: minute-level and grid-level output from preprocessor.py
PREPROCESSOR_OUTPUT_DIR = r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Class Meeting Downing Soc 220 2⧸4⧸2021/preprocessor_output_now"
# Output one scoring file every 3 minutes:
# engagement_0m_3m.json, engagement_3m_6m.json, ...
OUTPUT_DIR = r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Class Meeting Downing Soc 220 2⧸4⧸2021/focus_output_now"
WINDOW_MINUTES = 3


def get_minute_files():
    """Find all preprocessor minute files and sort by minute index."""
    if not os.path.isdir(PREPROCESSOR_OUTPUT_DIR):
        return []
    out = []
    for f in os.listdir(PREPROCESSOR_OUTPUT_DIR):
        if not f.endswith(".json") or not f.startswith("minute_"):
            continue
        try:
            num = int(f.replace("minute_", "").replace(".json", ""))
            out.append((num, f))
        except ValueError:
            continue
    return [fname for _, fname in sorted(out, key=lambda x: x[0])]


def minute_index_from_fname(fname: str) -> int:
    """minute_00.json -> 0, minute_03.json -> 3"""
    try:
        return int(fname.replace("minute_", "").replace(".json", ""))
    except Exception:
        return 0


def group_minutes_by_window(minute_fnames, window_minutes=3):
    """
    Build windows every `window_minutes` minutes.
    Return [(window_label, [fname1, fname2, ...]), ...]
    Example: (0m_3m, [minute_00, minute_01, minute_02]), (3m_6m, [minute_03, 04, 05]), ...
    """
    if not minute_fnames:
        return []
    with_idx = [(minute_index_from_fname(f), f) for f in minute_fnames]
    with_idx.sort(key=lambda x: x[0])
    minute_fnames = [f for _, f in with_idx]
    groups = []
    i = 0
    while i < len(minute_fnames):
        window_files = minute_fnames[i : i + window_minutes]
        if not window_files:
            break
        start_min = minute_index_from_fname(window_files[0])
        end_min = minute_index_from_fname(window_files[-1]) + 1
        window_label = f"{start_min}m_{end_min}m"
        groups.append((window_label, window_files))
        i += window_minutes
    return groups

# ENGAGEMENT PROMPT (TEXT-BASED)
ENGAGEMENT_PROMPT = """
You are a behavioral engagement evaluation model.

You are given observational evidence (scene descriptions from visual frames) describing a person's
visible behavior across a 3-minute window (three 1-minute segments, one paragraph per minute).

The evidence is standardized narrative from preprocessor output (one description per minute per grid).

Your task is to assess the engagement level of the person
based strictly on the observable evidence provided.

Engagement is defined on FOUR ordered levels:

0 - Not Engaged:
- frequent looking away or down
- obvious inattentive behavior

1 - Barely Engaged:
- weak or unstable attention
- inconsistent orientation or posture

2 - Engaged:
- generally oriented forward
- stable posture most of the time
- attentive with minor lapses

3 - Highly Engaged:
- this is the DEFAULT level when evidence shows
  stable forward orientation and posture
- no dominant or sustained distraction
- brief lapses are acceptable

You must make a holistic judgment based on the OVERALL tendency
across all segments.


Strict calibration policy:
- Do not avoid low scores. Use 0 and 1 whenever evidence supports them.
- Do not avoid high scores. Use 3 whenever strong forward/stable engagement is evident.
- Do NOT use Level 2 as a safe middle ground. If the person shows recurring distractions or "passive" looking, you can consider drop to Level 1.
- If deciding between 2 and 3, lean toward 3 when there is no strong evidence of sustained distraction.
- If deciding between 1 and 2, use 1 when attention is unstable or often not forward.


---

### ORDINAL JUDGMENT INSTRUCTIONS (IMPORTANT)

Based on the definitions above, first choose the SINGLE engagement
level (0–3) that best represents the overall behavior.
This will be referred to as the ANCHOR level.

Then, assess whether neighboring engagement levels could still be
reasonable interpretations of the same evidence, considering the
ORDER of the levels.

Specifically, judge the following ordinal deviations from the anchor:

- Is a ONE-LEVEL LOWER interpretation (anchor - 1) still plausible?
- Is a ONE-LEVEL HIGHER interpretation (anchor + 1) still plausible?
- Is a TWO-LEVEL LOWER interpretation (anchor - 2) plausible?
- Is a TWO-LEVEL HIGHER interpretation (anchor + 2) plausible?

These judgments must respect ordinal distance:
- A two-level deviation should NOT be considered plausible
  if the corresponding one-level deviation is not plausible.
- Larger deviations should be treated as less plausible
  than smaller deviations.

Do NOT assign probabilities or confidence scores directly.
Only make binary plausibility judgments for each deviation.

---

Return STRICT JSON ONLY:

{{
  "anchor": int,
  "acceptable_deviation": {{
    "-1": bool,
    "+1": bool,
    "-2": bool,
    "+2": bool
  }},
  "reasoning": "Concise explanation grounded in the evidence"
}}

## ATTENTION_EVIDENCE
{attention_text}

"""


def reconstruct_confidence_from_ordinal(anchor, acceptable_deviation, alpha=1.0):
    """
    Sakai-style ordinal-consistent confidence reconstruction.
    anchor: int (0-3)
    acceptable_deviation: dict with keys "-1","+1","-2","+2"
    """
    MAX_DISTANCE = 3
    distances = {}

    for label in range(4):
        d = abs(label - anchor)

        if d == 0:
            distances[label] = 0
        elif d == 1:
            key = "+1" if label > anchor else "-1"
            distances[label] = 1 if acceptable_deviation.get(key, False) else MAX_DISTANCE
        elif d == 2:
            key = "+2" if label > anchor else "-2"
            distances[label] = 2 if acceptable_deviation.get(key, False) else MAX_DISTANCE
        else:
            distances[label] = MAX_DISTANCE

        # enforce monotonic ordinal consistency
        distances[label] = max(distances[label], d)

    # convert distances -> weights (exponential decay)
    weights = {k: math.exp(-alpha * distances[k]) for k in distances}
    total = sum(weights.values())

    if total <= 0:
        raise ValueError("Invalid ordinal weights")

    confidence = {str(k): weights[k] / total for k in weights}
    return confidence

def choose_by_expected_mae(confidence_dict):
    probs = [confidence_dict[str(i)] for i in range(4)]
    best_k = None
    best_loss = float("inf")

    for k in range(4):
        loss = sum(abs(k - i) * probs[i] for i in range(4))
        if loss < best_loss:
            best_loss = loss
            best_k = k

    return best_k

def call_llm(prompt: str) -> dict:
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, REQUEST_COUNT

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()

    # Support responses wrapped in ```json fences
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    usage = getattr(response, "usage", None)
    with USAGE_LOCK:
        if usage is not None:
            TOTAL_PROMPT_TOKENS += getattr(usage, "prompt_tokens", 0)
            TOTAL_COMPLETION_TOKENS += getattr(usage, "completion_tokens", 0)
        REQUEST_COUNT += 1

    return json.loads(content)


def score_single_grid(grid_id: str, minute_data: list, alpha: float = 0.7):
    blocks = []
    for i, data in enumerate(minute_data):
        if grid_id not in data:
            continue
        text = data[grid_id]
        if isinstance(text, str) and text.strip():
            blocks.append(f"[MINUTE {i+1}]\n{text.strip()}")

    if not blocks:
        return grid_id, None, None

    evidence_text = "\n\n".join(blocks)
    prompt = ENGAGEMENT_PROMPT.format(attention_text=evidence_text)

    try:
        result = call_llm(prompt)
    except Exception as e:
        return grid_id, {"engagement": None, "reasoning": str(e)}, None

    anchor = result["anchor"]
    acceptable_deviation = result["acceptable_deviation"]

    confidence = reconstruct_confidence_from_ordinal(
        anchor=anchor,
        acceptable_deviation=acceptable_deviation,
        alpha=alpha
    )

    engagement = choose_by_expected_mae(confidence)

    payload = {
        "engagement": engagement,
        "confidence": confidence,
        "anchor": anchor,
        "acceptable_deviation": acceptable_deviation,
        "reasoning": result["reasoning"],
        "num_segments": len(blocks)
    }
    return grid_id, payload, engagement



def main():
    run_start_ts = time.time()
    window_metrics = []
    total_candidate_grids = 0
    total_success_grids = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    minute_fnames = get_minute_files()
    if not minute_fnames:
        raise FileNotFoundError(f"No minute_*.json found in {PREPROCESSOR_OUTPUT_DIR}")

    windows = group_minutes_by_window(minute_fnames, WINDOW_MINUTES)
    if not windows:
        print("No 3-minute windows to process.")
        return

    for window_label, window_fnames in windows:
        window_start_ts = time.time()
        print(f"\n[WINDOW] {window_label} from {window_fnames}")

        minute_data = []
        for fname in window_fnames:
            path = os.path.join(PREPROCESSOR_OUTPUT_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                minute_data.append(json.load(f))

        # If all minute_*.json files in this 3-minute window have no non-empty text,
        # treat it as a PPT lecture (no usable student video evidence) and skip scoring output.
        has_any_evidence = False
        for data in minute_data:
            if not isinstance(data, dict):
                continue
            for v in data.values():
                if isinstance(v, str) and v.strip():
                    has_any_evidence = True
                    break
            if has_any_evidence:
                break

        if not has_any_evidence:
            print(f"[SKIP] {window_label}: no preprocessor output (likely PPT lecture), skip engagement scoring.")
            continue

        grid_ids = sorted(set().union(*(set(d.keys()) for d in minute_data)))
        results = {}
        total_candidate_grids += len(grid_ids)

        if grid_ids:
            max_workers = max(1, min(MAX_WORKERS, len(grid_ids)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(score_single_grid, grid_id, minute_data): grid_id
                    for grid_id in grid_ids
                }
                for future in as_completed(futures):
                    grid_id = futures[future]
                    try:
                        done_grid_id, payload, engagement = future.result()
                    except Exception as e:
                        results[grid_id] = {
                            "engagement": None,
                            "reasoning": f"thread failed: {e}"
                        }
                        continue

                    if payload is None:
                        continue

                    results[done_grid_id] = payload
                    print(f"[DONE] {window_label} {done_grid_id} -> {engagement}")

        out_path = os.path.join(OUTPUT_DIR, f"engagement_{window_label}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            ordered_results = {k: results[k] for k in sorted(results.keys())}
            json.dump(ordered_results, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")
        total_success_grids += len(results)
        window_elapsed = time.time() - window_start_ts
        window_metrics.append(
            {
                "window_label": window_label,
                "candidate_grids": len(grid_ids),
                "successful_grids": len(results),
                "elapsed_sec": round(window_elapsed, 3),
            }
        )
        print(
            f"[WINDOW_METRIC] {window_label}: success={len(results)}/{len(grid_ids)}, "
            f"elapsed={window_elapsed:.2f}s"
        )

    run_elapsed = time.time() - run_start_ts
    total_tokens = TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS
    prompt_cost = (TOTAL_PROMPT_TOKENS / 1_000_000) * PROMPT_COST_PER_1M
    completion_cost = (TOTAL_COMPLETION_TOKENS / 1_000_000) * COMPLETION_COST_PER_1M
    total_cost = prompt_cost + completion_cost

    if REQUEST_COUNT > 0:
        avg_prompt = TOTAL_PROMPT_TOKENS / REQUEST_COUNT
        avg_completion = TOTAL_COMPLETION_TOKENS / REQUEST_COUNT
        req_per_sec = REQUEST_COUNT / max(1e-9, run_elapsed)
        token_per_sec = total_tokens / max(1e-9, run_elapsed)
        print(
            f"\n[USAGE] requests={REQUEST_COUNT}, "
            f"total_prompt_tokens={TOTAL_PROMPT_TOKENS}, "
            f"total_completion_tokens={TOTAL_COMPLETION_TOKENS}, "
            f"avg_prompt_tokens_per_req={avg_prompt:.1f}, "
            f"avg_completion_tokens_per_req={avg_completion:.1f}, "
            f"elapsed_sec={run_elapsed:.2f}, "
            f"req_per_sec={req_per_sec:.3f}, "
            f"tokens_per_sec={token_per_sec:.1f}"
        )
        print(
            f"[COST] prompt=${prompt_cost:.4f}, completion=${completion_cost:.4f}, total=${total_cost:.4f}"
        )

    efficiency_report = {
        "model": MODEL,
        "max_workers": MAX_WORKERS,
        "elapsed_sec": round(run_elapsed, 3),
        "requests": REQUEST_COUNT,
        "total_prompt_tokens": TOTAL_PROMPT_TOKENS,
        "total_completion_tokens": TOTAL_COMPLETION_TOKENS,
        "total_tokens": total_tokens,
        "req_per_sec": round(REQUEST_COUNT / max(1e-9, run_elapsed), 6),
        "tokens_per_sec": round(total_tokens / max(1e-9, run_elapsed), 3),
        "prompt_cost_per_1m": PROMPT_COST_PER_1M,
        "completion_cost_per_1m": COMPLETION_COST_PER_1M,
        "estimated_prompt_cost_usd": round(prompt_cost, 6),
        "estimated_completion_cost_usd": round(completion_cost, 6),
        "estimated_total_cost_usd": round(total_cost, 6),
        "total_candidate_grids": total_candidate_grids,
        "total_success_grids": total_success_grids,
        "success_ratio": round(total_success_grids / max(1, total_candidate_grids), 6),
        "window_metrics": window_metrics,
    }
    report_path = os.path.join(OUTPUT_DIR, "focus_scoring_efficiency_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(efficiency_report, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {report_path}")


if __name__ == "__main__":
    main()
