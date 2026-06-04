from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional


ROOT = Path("")
STUDENT_SCRIPT = 
TEACHER_SCRIPT = 
EXPECT_SCRIPT = 
SUGGESTION_SCRIPT = 


def load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_project_root(segments_json: Path) -> Optional[Path]:
    parts = list(segments_json.parts)
    if "icap_output" not in parts:
        return None
    idx = parts.index("icap_output")
    return Path(*parts[:idx])


def infer_default_output_dir(segments_json: Path) -> Path:
    root = infer_project_root(segments_json)
    if root is not None:
        return root / "icap_output" / "pipeline_auto"
    return segments_json.parent / "pipeline_auto"


def infer_default_writing_behavior(segments_json: Path) -> Optional[Path]:
    root = infer_project_root(segments_json)
    if root is None:
        return None
    candidate = root / "writing_behavior.json"
    if candidate.exists():
        return candidate
    return None


def extract_efficiency(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = load_json(path)
    if "efficiency_summary" in payload:
        return payload.get("efficiency_summary", {})
    return payload.get("efficiency", {})


def run_student(
    segments_json: Path,
    output_dir: Path,
    model: Optional[str],
    max_workers: Optional[int],
) -> Path:
    module = load_module(STUDENT_SCRIPT, "icap_student_runtime")
    module.INPUT_JSON = segments_json
    module.OUTPUT_DIR = output_dir
    if model:
        module.MODEL = model
    if max_workers is not None and max_workers > 0:
        module.MAX_WORKERS = max_workers
    module.main()
    return output_dir / "knowledge_segments_global_classification.json"


def run_teacher(
    segments_json: Path,
    output_dir: Path,
    model: Optional[str],
    max_workers: Optional[int],
) -> Path:
    module = load_module(TEACHER_SCRIPT, "icap_teacher_runtime")
    teacher_json = output_dir / "teacher_current_ci_by_knowledge.json"
    teacher_md = output_dir / "teacher_current_ci_by_knowledge.md"
    argv = [
        str(TEACHER_SCRIPT),
        "--input-json",
        str(segments_json),
        "--output-json",
        str(teacher_json),
        "--output-md",
        str(teacher_md),
    ]
    if model:
        argv.extend(["--model", model])
    if max_workers is not None and max_workers > 0:
        argv.extend(["--max-workers", str(max_workers)])
    old_argv = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old_argv
    return teacher_json


def run_expect(
    segments_json: Path,
    student_json: Path,
    output_dir: Path,
    model: Optional[str],
    max_workers: Optional[int],
) -> Path:
    module = load_module(EXPECT_SCRIPT, "icap_expect_runtime")
    module.KNOWLEDGE_JSON = segments_json
    module.STUDENT_CI_JSON = student_json
    module.OUTPUT_JSON = output_dir / "teacher_expected_ci_by_knowledge.json"
    module.OUTPUT_MD = output_dir / "teacher_expected_ci_by_knowledge.md"
    if model:
        module.MODEL = model
    if max_workers is not None and max_workers > 0:
        module.MAX_WORKERS = max_workers
    module.main()
    return module.OUTPUT_JSON


def run_suggestion(
    segments_json: Path,
    writing_behavior_json: Path,
    student_json: Path,
    teacher_json: Path,
    expect_json: Path,
    output_dir: Path,
    class_size: int,
    model: Optional[str],
    max_workers: Optional[int],
) -> Path:
    module = load_module(SUGGESTION_SCRIPT, "icap_suggestion_runtime")
    module.PATHS = {
        "writing": writing_behavior_json,
        "knowledge": segments_json,
        "student_ci": student_json,
        "teacher_curr": teacher_json,
        "teacher_exp": expect_json,
        "out_dir": output_dir,
    }
    if model:
        module.MODEL = model
    if max_workers is not None and max_workers > 0:
        module.MAX_WORKERS = max_workers

    old_argv = sys.argv
    try:
        sys.argv = [str(SUGGESTION_SCRIPT), "--class-size", str(class_size)]
        module.main()
    finally:
        sys.argv = old_argv
    return output_dir / "icap_suggestions.json"


def resolve_existing_upstream_json(
    output_dir: Path,
    student_json: Optional[Path],
    teacher_json: Optional[Path],
    expect_json: Optional[Path],
) -> Dict[str, Path]:
    resolved_student = (
        student_json.resolve()
        if student_json is not None
        else (output_dir / "knowledge_segments_global_classification.json")
    )
    resolved_teacher = (
        teacher_json.resolve()
        if teacher_json is not None
        else (output_dir / "teacher_current_ci_by_knowledge.json")
    )
    resolved_expect = (
        expect_json.resolve()
        if expect_json is not None
        else (output_dir / "teacher_expected_ci_by_knowledge.json")
    )
    for p in [resolved_student, resolved_teacher, resolved_expect]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required upstream result for --only-suggestion: {p}"
            )
    return {
        "student_json": resolved_student,
        "teacher_json": resolved_teacher,
        "expect_json": resolved_expect,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ICAP teacher pipeline from one knowledge-segments JSON."
    )
    parser.add_argument(
        "--segments-json",
        required=True,
        type=Path,
        help="Path to knowledge_segments_global*.json",
    )
    parser.add_argument(
        "--writing-behavior-json",
        type=Path,
        default=None,
        help="Optional writing_behavior.json path (auto-infer if omitted).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save all outputs (default: inferred pipeline_auto dir).",
    )
    parser.add_argument(
        "--class-size",
        type=int,
        default=30,
        help="Class size for icap_suggestion.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override for pipeline steps.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional max workers override for pipeline steps.",
    )
    parser.add_argument(
        "--only-suggestion",
        action="store_true",
        help="Only run icap_suggestion.py (reuse existing student/teacher/expect outputs).",
    )
    parser.add_argument(
        "--student-json",
        type=Path,
        default=None,
        help="Optional existing student classification JSON path for --only-suggestion mode.",
    )
    parser.add_argument(
        "--teacher-json",
        type=Path,
        default=None,
        help="Optional existing teacher current CI JSON path for --only-suggestion mode.",
    )
    parser.add_argument(
        "--expect-json",
        type=Path,
        default=None,
        help="Optional existing teacher expected CI JSON path for --only-suggestion mode.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional output summary file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segments_json = args.segments_json.resolve()
    if not segments_json.exists():
        raise FileNotFoundError(f"segments_json not found: {segments_json}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else infer_default_output_dir(segments_json).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.writing_behavior_json is not None:
        writing_behavior_json = args.writing_behavior_json.resolve()
    else:
        inferred = infer_default_writing_behavior(segments_json)
        if inferred is None:
            raise ValueError(
                "Cannot infer writing_behavior.json. Please pass --writing-behavior-json."
            )
        writing_behavior_json = inferred.resolve()
    if not writing_behavior_json.exists():
        raise FileNotFoundError(
            f"writing_behavior_json not found: {writing_behavior_json}"
        )

    print(f"[START] segments_json={segments_json}")
    print(f"[START] writing_behavior_json={writing_behavior_json}")
    print(f"[START] output_dir={output_dir}")

    pipeline_start = time.perf_counter()
    step_times: Dict[str, float] = {}

    if args.only_suggestion:
        reused = resolve_existing_upstream_json(
            output_dir=output_dir,
            student_json=args.student_json,
            teacher_json=args.teacher_json,
            expect_json=args.expect_json,
        )
        student_json = reused["student_json"]
        teacher_json = reused["teacher_json"]
        expect_json = reused["expect_json"]
        print("[INFO] only-suggestion mode: reusing existing upstream JSON files.")
    else:
        t0 = time.perf_counter()
        student_json = run_student(
            segments_json=segments_json,
            output_dir=output_dir,
            model=args.model,
            max_workers=args.max_workers,
        )
        step_times["icap_student_sec"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        teacher_json = run_teacher(
            segments_json=segments_json,
            output_dir=output_dir,
            model=args.model,
            max_workers=args.max_workers,
        )
        step_times["icap_teacher_sec"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        expect_json = run_expect(
            segments_json=segments_json,
            student_json=student_json,
            output_dir=output_dir,
            model=args.model,
            max_workers=args.max_workers,
        )
        step_times["icap_expect_sec"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    suggestion_json = run_suggestion(
        segments_json=segments_json,
        writing_behavior_json=writing_behavior_json,
        student_json=student_json,
        teacher_json=teacher_json,
        expect_json=expect_json,
        output_dir=output_dir,
        class_size=args.class_size,
        model=args.model,
        max_workers=args.max_workers,
    )
    step_times["icap_suggestion_sec"] = round(time.perf_counter() - t0, 3)

    outputs: Dict[str, str] = {
        "student_json": str(student_json),
        "teacher_json": str(teacher_json),
        "expect_json": str(expect_json),
        "suggestion_json": str(suggestion_json),
    }

    efficiency_from_steps: Dict[str, Dict[str, Any]] = {
        "student": extract_efficiency(student_json),
        "teacher": extract_efficiency(teacher_json),
        "expect": extract_efficiency(expect_json),
        "suggestion": extract_efficiency(suggestion_json),
    }

    summary = {
        "inputs": {
            "segments_json": str(segments_json),
            "writing_behavior_json": str(writing_behavior_json),
        },
        "outputs": outputs,
        "config": {
            "class_size": args.class_size,
            "model_override": args.model,
            "max_workers_override": args.max_workers,
            "only_suggestion": args.only_suggestion,
        },
        "step_elapsed_sec": step_times,
        "total_elapsed_sec": round(time.perf_counter() - pipeline_start, 3),
        "efficiency_from_steps": efficiency_from_steps,
    }

    summary_path = (
        args.summary_json.resolve()
        if args.summary_json
        else (output_dir / "pipeline_run_summary.json")
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] pipeline finished")
    print(f"[DONE] student: {student_json}")
    print(f"[DONE] teacher: {teacher_json}")
    print(f"[DONE] expect: {expect_json}")
    print(f"[DONE] suggestion: {suggestion_json}")
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
# python "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/run_pipeline.py" \
#   --segments-json "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Meeting for 3D Design/icap_output/icap_split_free/knowledge_segments_global_free.json" \
#   --writing-behavior-json "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Meeting for 3D Design/writing_behavior.json" \
#   --output-dir "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/backbone_sensitivity/claude-4.5-sonnet/stem2" \
#   --model "anthropic/claude-4.5-sonnet"
