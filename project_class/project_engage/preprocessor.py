'''
[USAGE] requests=1500, total_prompt_tokens=76879500, 
total_completion_tokens=322774, avg_prompt_tokens_per_req=51253.0, 
avg_completion_tokens_per_req=215.2
≈ $11.72 USD

[USAGE] requests=1353, total_prompt_tokens=66032212, total_completion_tokens=296806, 
avg_prompt_tokens_per_req=48804.3, avg_completion_tokens_per_req=219.4
Total estimated cost ≈ 9.90 + 0.18 ≈ $10.08 USD

[USAGE] requests=598, total_prompt_tokens=28583666, 
total_completion_tokens=115671, avg_prompt_tokens_per_req=47798.8, avg_completion_tokens_per_req=193.4

[USAGE] requests=681, total_prompt_tokens=33241619, total_completion_tokens=130850, avg_prompt_tokens_per_req=48813.0, avg_completion_tokens_per_req=192.1, elapsed_sec=488.41, req_per_sec=1.394, tokens_per_sec=68328.7
[COST] prompt=$4.9862, completion=$0.0785, total=$5.0648
'''
import os
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


# Initialize OpenAI client
TIMEOUT = 60
# Parallel API calls per minute across grids.
# Increase MAX_WORKERS for higher throughput, decrease if you hit rate limits.
MAX_WORKERS = 20
# Cost model (USD per 1M tokens). Adjust if your provider pricing changes.
PROMPT_COST_PER_1M = 0.15
COMPLETION_COST_PER_1M = 0.60

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

IMAGE_ROOT = r""
OUTPUT_DIR = r""
PPT_EXCLUDED_TIMES_JSON = r""
ICON_EXCLUDED_FRAMES_JSON = r""
BLACK_EXCLUDED_FRAMES_JSON = r""
# Resume controls:
# - Set START_MINUTE=17 to continue from minute_17.
# - Set to None to process all minutes from the beginning.
START_MINUTE = 0
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
REQUEST_COUNT = 0
USAGE_LOCK = threading.Lock()


# PROMPT
PREPROCESSOR_PROMPT = """
You are a visual scene narrator specializing in human behavior and body language analysis.

You will be shown several image frames from a video. Your task is to provide a detailed third-person description of the scene, focusing specifically on the person's posture, gestures, and behavioral cues.

Observation Focus (Pay close attention to these features):**
- Posture & Movement**: Detect if arms are crossed, slouching, hand stretching, or if there is a consistent pose vs. changing seating positions. Note if the person moves closer to the screen or tilts their head towards it.
- Hand Gestures**: Identify active hand movements, hands playing with objects, hands at the back of the head, or a hand on the mouth (thinking pose). Note if the person is modifying their glasses.
- Facial & Oral Actions**: Detect nodding, yawning, smiling, speaking, or drinking/eating.
- *emporal Changes**: Note any "Sudden behavior change" between frames, or if the person has just returned to the scene (back from a break).

Critical Action Detection**:
- Describe the presence of writing instruments (pens, pencils, styluses).
- If you are VERY SURE a writing action is occurring, add a **"writing"** tag to your description.

Guidelines**:
- Describe ONLY what is visibly shown. Do NOT infer emotions, intentions, or mental states (e.g., instead of "bored," say "yawning and slouching").
- Use natural, continuous prose.
- If the person is still, describe the stillness and the specific pose they are maintaining.

Style:
- Neutral, objective, and descriptive.
- Third-person narration.
- 1–3 short, information-dense paragraphs.
"""

# UTILS

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# CORE FUNCTION (process multiple frames for one grid within one minute at once)

def run_minute_frames_to_text(minute_id: str, grid_id: str, image_paths):
    """
    Input: 1–6 frame paths from the same 1-minute window and grid
    (after cleaning there may be fewer than 6 frames; logic is compatible).
    Output: one detailed natural language description for the WHOLE minute (string).
    """
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, REQUEST_COUNT

    if not image_paths:
        return None, "no images"

    # There may be fewer than 6 frames after cleaning; use available frames (up to 6).
    image_paths = image_paths[:6]
    n_frames = len(image_paths)

    intro_text = (
        PREPROCESSOR_PROMPT
        + "\n\nYou are given "
        + (f"{n_frames} frame(s) " if n_frames < 6 else "up to six frames ")
        + "from the same 1-minute window "
        + f"(folder: {minute_id}, grid: {grid_id}). "
        + "Write ONE coherent description for the entire minute, taking into account "
        + "what is common and what changes across these frames."
    )

    user_content = [{"type": "text", "text": intro_text}]
    for p in image_paths:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(p)}"},
            }
        )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
            timeout=TIMEOUT,
        )
        text = (response.choices[0].message.content or "").strip()

        usage = getattr(response, "usage", None)
        with USAGE_LOCK:
            if usage is not None:
                TOTAL_PROMPT_TOKENS += getattr(usage, "prompt_tokens", 0)
                TOTAL_COMPLETION_TOKENS += getattr(usage, "completion_tokens", 0)
            REQUEST_COUNT += 1
    except Exception as e:
        return None, f"Request failed: {e}"

    return text, None


# MAIN PIPELINE

def _load_ppt_excluded_times() -> dict:
    """If ppt_excluded_times.json exists, return {minute_id: set(frame_id)}; otherwise return {}."""
    if not os.path.isfile(PPT_EXCLUDED_TIMES_JSON):
        return {}
    try:
        with open(PPT_EXCLUDED_TIMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: set(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_icon_excluded_frames() -> dict:
    """If icon_excluded_frames.json exists, return {minute_id: {grid_id: set(frame_id)}}; otherwise return {}."""
    if not os.path.isfile(ICON_EXCLUDED_FRAMES_JSON):
        return {}
    try:
        with open(ICON_EXCLUDED_FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {mid: {gid: set(flist) for gid, flist in grids.items()} for mid, grids in data.items()}
    except Exception:
        return {}


def _load_black_excluded_frames() -> dict:
    """If black_excluded_frames.json exists, return {minute_id: {grid_id: set(frame_id)}}; otherwise return {}."""
    if not os.path.isfile(BLACK_EXCLUDED_FRAMES_JSON):
        return {}
    try:
        with open(BLACK_EXCLUDED_FRAMES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {mid: {gid: set(flist) for gid, flist in grids.items()} for mid, grids in data.items()}
    except Exception:
        return {}


def main():
    run_start_ts = time.time()
    minute_metrics = []
    total_candidate_grids = 0
    total_success_grids = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ppt_excluded_times = _load_ppt_excluded_times()
    icon_excluded_frames = _load_icon_excluded_frames()
    black_excluded_frames = _load_black_excluded_frames()
    if ppt_excluded_times:
        n = sum(len(v) for v in ppt_excluded_times.values())
        print(f"[PPT] Excluding {n} time slots (grid_00=PPT) from all grids")
    if icon_excluded_frames:
        n = sum(len(v) for g in icon_excluded_frames.values() for v in g.values())
        print(f"[ICON] Excluding {n} single frames (icon per grid)")
    if black_excluded_frames:
        n = sum(len(v) for g in black_excluded_frames.values() for v in g.values())
        print(f"[BLACK] Excluding {n} single frames (black per grid)")

    for minute_id in sorted(os.listdir(IMAGE_ROOT)):
        minute_start_ts = time.time()
        video_dir = os.path.join(IMAGE_ROOT, minute_id)
        if not os.path.isdir(video_dir):
            continue
        if not minute_id.startswith("minute_"):
            continue

        try:
            minute_num = int(minute_id.replace("minute_", ""))
        except ValueError:
            continue

        if START_MINUTE is not None and minute_num < START_MINUTE:
            continue

        print(f"\n[MINUTE] {minute_id}")
        video_outputs = {}
        ppt_frames = ppt_excluded_times.get(minute_id, set())
        icon_grid_frames = icon_excluded_frames.get(minute_id, {})
        black_grid_frames = black_excluded_frames.get(minute_id, {})

        grid_inputs = []
        for grid_id in sorted(os.listdir(video_dir)):
            grid_dir = os.path.join(video_dir, grid_id)
            if not os.path.isdir(grid_dir):
                continue

            all_paths = sorted(
                [
                    os.path.join(grid_dir, f)
                    for f in os.listdir(grid_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            icon_frames = icon_grid_frames.get(grid_id, set())
            black_frames = black_grid_frames.get(grid_id, set())
            # Exclude: frames removed by PPT at this timestamp (all grids)
            #        + icon-filtered frames for this grid
            #        + black-frame-filtered frames for this grid
            image_paths = [
                p
                for p in all_paths
                if (frame_id := os.path.splitext(os.path.basename(p))[0]) not in ppt_frames
                and frame_id not in icon_frames
                and frame_id not in black_frames
            ]

            if not image_paths:
                continue

            grid_inputs.append((grid_id, image_paths))

        if grid_inputs:
            total_candidate_grids += len(grid_inputs)
            max_workers = max(1, min(MAX_WORKERS, len(grid_inputs)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_minute_frames_to_text, minute_id, grid_id, image_paths): grid_id
                    for grid_id, image_paths in grid_inputs
                }
                completed_results = {}
                for future in as_completed(futures):
                    grid_id = futures[future]
                    try:
                        text, err = future.result()
                    except Exception as e:
                        text, err = None, f"thread failed: {e}"

                    if err or not text:
                        continue
                    lower = text.lower()
                    if lower.startswith(("i'm sorry", "i’m sorry", "i cannot", "i can't")):
                        continue
                    completed_results[grid_id] = text

            for grid_id in sorted(completed_results.keys()):
                # Output only one minute-level narrative per grid
                video_outputs[grid_id] = completed_results[grid_id]
                print(f"[OK] {grid_id}: 1 minute-level description")

            total_success_grids += len(completed_results)

        if video_outputs:
            out_path = os.path.join(OUTPUT_DIR, f"{minute_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(video_outputs, f, indent=2, ensure_ascii=False)
            print(f"[SAVED] {out_path}")

        minute_elapsed = time.time() - minute_start_ts
        minute_metrics.append(
            {
                "minute_id": minute_id,
                "candidate_grids": len(grid_inputs),
                "successful_grids": len(video_outputs),
                "elapsed_sec": round(minute_elapsed, 3),
            }
        )
        if len(grid_inputs) > 0:
            print(
                f"[MINUTE_METRIC] {minute_id}: "
                f"success={len(video_outputs)}/{len(grid_inputs)}, "
                f"elapsed={minute_elapsed:.2f}s"
            )

    # Print basic token usage and efficiency stats
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
        "minute_metrics": minute_metrics,
    }
    report_path = os.path.join(OUTPUT_DIR, "preprocessor_efficiency_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(efficiency_report, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {report_path}")


if __name__ == "__main__":
    main()
