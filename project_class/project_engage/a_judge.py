import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI


INPUT_DIR = Path("")
OUTPUT_JSON = Path("")
MODEL = "gpt-4o-mini"
TIMEOUT = 60
MAX_WORKERS = 10


def build_client() -> OpenAI:
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if openrouter_key:
        return OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )
    if openai_key:
        return OpenAI(api_key=openai_key, timeout=TIMEOUT)

    raise ValueError(
        "API key is missing. Please set OPENROUTER_API_KEY or OPENAI_API_KEY."
    )


def parse_minute(file_name: str) -> int:
    return int(file_name.replace("minute_", "").replace(".json", ""))


def normalize_json_text(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    return content


def build_prompt(description: str) -> str:
    return f"""
You are a strict behavior classifier.
Determine whether this classroom visual description contains POSITIVE evidence of writing behavior.

Writing behavior includes:
- actively writing notes/answers
- holding pen/pencil/stylus and writing
- note-taking/drawing for class task

If description says "no writing", "no writing detected", "not clearly visible", or uncertain mention only,
then has_writing_behavior must be false.

Return STRICT JSON only:
{{
  "has_writing_behavior": true/false,
  "reasoning": "one concise sentence in English"
}}

Description:
{description}
"""


def classify_writing_behavior(client: OpenAI, description: str) -> Tuple[bool, str]:
    if not isinstance(description, str) or not description.strip():
        return False, "Empty description."

    prompt = build_prompt(description.strip())
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(normalize_json_text(content))
    return bool(payload.get("has_writing_behavior", False)), str(payload.get("reasoning", "")).strip()


def classify_single_grid(
    client: OpenAI, minute: int, minute_file: Path, grid_id: str, desc: str
) -> Dict:
    try:
        ok, reason = classify_writing_behavior(client, desc if isinstance(desc, str) else "")
    except Exception as e:
        ok, reason = False, f"LLM error: {e}"

    if not ok:
        return {}

    return {
        "minute": minute,
        "grid": grid_id,
        "reasoning": reason,
        "file": minute_file.name,
    }


def scan_all_minutes(input_dir: Path, client: OpenAI) -> List[Dict]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    minute_files = sorted(input_dir.glob("minute_*.json"), key=lambda p: parse_minute(p.name))
    hits: List[Dict] = []
    tasks: List[Tuple[int, Path, str, str]] = []

    for minute_file in minute_files:
        minute = parse_minute(minute_file.name)
        with open(minute_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        for grid_id, desc in data.items():
            tasks.append((minute, minute_file, grid_id, desc if isinstance(desc, str) else ""))

    if not tasks:
        return hits

    max_workers = max(1, min(MAX_WORKERS, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(classify_single_grid, client, minute, minute_file, grid_id, desc): (
                minute,
                grid_id,
            )
            for minute, minute_file, grid_id, desc in tasks
        }
        for future in as_completed(futures):
            minute, grid_id = futures[future]
            try:
                payload = future.result()
            except Exception as e:
                print(f"[ERR] minute={minute}, grid={grid_id}, error={e}")
                continue
            if not payload:
                continue
            hits.append(payload)
            print(f"[HIT] minute={payload['minute']}, grid={payload['grid']}, reasoning={payload['reasoning']}")

    hits.sort(key=lambda x: (x["minute"], x["grid"]))

    return hits


def main() -> None:
    client = build_client()
    hits = scan_all_minutes(INPUT_DIR, client)
    summary = {
        "model": MODEL,
        "input_dir": str(INPUT_DIR),
        "total_hits": len(hits),
        "hits": hits,
    }

    print(f"Total writing behavior hits: {len(hits)}")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
