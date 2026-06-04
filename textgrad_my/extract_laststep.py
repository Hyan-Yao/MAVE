from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_INPUT = Path(
    ""
)
DEFAULT_OUTPUT = Path(
    ""
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract last-step text_gradient for each knowledge point from JSONL."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    return parser.parse_args()


def parse_kp_id(row: Dict[str, Any]) -> Optional[int]:
    src = str(row.get("input_transcript_path", ""))
    m = re.search(r"knowledge_point_id=(\d+)", src)
    if m:
        return int(m.group(1))
    w = str(row.get("window", ""))
    m = re.search(r"\bkp_(\d+)\b", w)
    if m:
        return int(m.group(1))
    return None


def choose_last_step_record(step_records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [r for r in step_records if isinstance(r, dict)]
    if not valid:
        return None
    try:
        return max(valid, key=lambda r: int(r.get("step", -1)))
    except Exception:
        return valid[-1]


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    results: List[Dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if not isinstance(row, dict):
                continue

            kp_id = parse_kp_id(row)
            if kp_id is None:
                continue

            step_records = row.get("step_records", [])
            if not isinstance(step_records, list):
                continue

            last = choose_last_step_record(step_records)
            if not last:
                continue

            text_gradient = str(last.get("text_gradient", "")).strip()
            if not text_gradient:
                continue

            step_value = last.get("step")
            try:
                step = int(step_value)
            except Exception:
                step = len(step_records) - 1

            results.append(
                {
                    "knowledge_point_id": kp_id,
                    "window": str(row.get("window", "")),
                    "input_transcript_path": str(row.get("input_transcript_path", "")),
                    "line_index": line_idx,
                    "last_step": step,
                    "text_gradient": text_gradient,
                }
            )

    results.sort(key=lambda x: x["knowledge_point_id"])
    payload = {
        "source_file": str(args.input),
        "count": len(results),
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Extracted {len(results)} records.")
    print(f"[DONE] Output: {args.output}")


if __name__ == "__main__":
    main()
