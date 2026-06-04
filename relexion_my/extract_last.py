import json
from pathlib import Path
from typing import Any, Dict, List


INPUT_JSONL = Path(
    ""
)
OUTPUT_JSON = Path(
    ""
)


def _group_key(rec: Dict[str, Any], line_idx: int) -> str:
    kp_id = rec.get("knowledge_point_id")
    if isinstance(kp_id, int):
        return f"kp_{kp_id}"
    window = rec.get("window")
    if isinstance(window, str) and window.strip():
        return f"window_{window.strip()}"
    return f"unknown_{line_idx}"


def _safe_step(rec: Dict[str, Any]) -> int:
    try:
        return int(rec.get("step", -1))
    except (TypeError, ValueError):
        return -1


def extract_last_steps(input_jsonl: Path) -> List[Dict[str, Any]]:
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl}")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            key = _group_key(rec, line_idx)
            grouped.setdefault(key, []).append(rec)

    out_rows: List[Dict[str, Any]] = []
    for _, rows in grouped.items():
        target = max(rows, key=_safe_step)
        out_rows.append(
            {
                "knowledge_point_id": target.get("knowledge_point_id"),
                "topic": target.get("topic", ""),
                "step": _safe_step(target),
                "strategy": target.get("strategy", ""),
                "reflection_signal": target.get("reflection_signal", ""),
            }
        )

    out_rows.sort(
        key=lambda r: (
            10**9 if not isinstance(r.get("knowledge_point_id"), int) else int(r["knowledge_point_id"]),
            int(r.get("step", -1)),
        )
    )
    return out_rows


def main() -> None:
    rows = extract_last_steps(INPUT_JSONL)
    payload = {
        "input_jsonl": str(INPUT_JSONL),
        "output_json": str(OUTPUT_JSON),
        "count": len(rows),
        "results": rows,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] extracted={len(rows)}")
    print(f"[OUT] {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
