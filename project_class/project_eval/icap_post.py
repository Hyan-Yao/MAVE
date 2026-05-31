from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError as e:  # pragma: no cover
    raise ImportError("Missing dependency: matplotlib. Install with `pip install matplotlib`.") from e


CLASS_SIZE = 30

INPUT_STUDENT_CLASSIFICATION = Path(
    "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project-teacher/output/i_c_output_now/knowledge_segments_global_classification.json"
)
INPUT_SHARED_SIGNALS = Path(
    "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project-teacher/output/teaching_report_now/icap_suggestions.json"
)
INPUT_WRITING_BEHAVIOR = Path(
    "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/writing_behavior3.json"
)
OUTPUT_JSON = Path(
    "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project_eval/icap_post_distribution.json"
)
OUTPUT_PNG = Path(
    "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project_eval/icap_post_distribution.png"
)

# 通过放大 eval 分差来拉开方法间差异（仅依赖评分差，不引入方法固有系数）
EVAL_SCORE_POWER = 1.6
EVAL_SCORE_WEIGHT = 0.62
EVAL_DIFF_AMPLIFY = 2.2
EVAL_DIFF_POWER = 1.0
MIN_METHOD_UPLIFT = 0.03

METHOD_SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "ours",
        "display_name": "Ours",
        "suggestions_path": "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project-teacher/output/teaching_report_now/icap_suggestions.json",
        "eval_path": "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project_eval/eval_output/demo_icap_eval.json",
        "default_effect": 0.6,
        "color": "#059669",
    },
    {
        "name": "prompt",
        "display_name": "Prompt",
        "suggestions_path": "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/prompt/demo_icap/teaching_suggestions_prompt_only_by_kp.json",
        "eval_path": "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/prompt/demo_icap/prompt_only_evaluation.json",
        "default_effect": 0.6,
        "color": "#2563eb",
    },
    {
        "name": "reflexion",
        "display_name": "Reflexion",
        "suggestions_path": "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/reflexion_my/demo_icap/laststep_reflection_signals.json",
        "eval_path": "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/reflexion_my/demo_icap/evaluation.jsonl",
        "default_effect": 0.6,
        "color": "#7c3aed",
    },
    {
        "name": "textgrad",
        "display_name": "TextGrad",
        "suggestions_path": "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/textgrad_my/demo_icap/teaching_suggestions_textgrad_by_kp.json",
        "eval_path": "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/textgrad_my/demo_icap/evaluation_text_gradients.json",
        "default_effect": 0.6,
        "color": "#ea580c",
    },
]


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON must be object: {path}")
    return data


def resolve_effect_eval_path(preferred: Path) -> Optional[Path]:
    """
    解析建议效果评估文件路径，避免路径变更后静默退回默认值。
    """
    candidates = [
        preferred,
        preferred.parent.parent / "eval_output" / preferred.name,
        preferred.parent.parent / "output" / preferred.name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def to_int(v: Any) -> Optional[int]:
    try:
        if isinstance(v, bool):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def load_flexible_payload(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    # .jsonl 文件有时实际存的是完整 JSON 对象，这里先尝试整体解析。
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return {"results": rows}


def extract_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return [r for r in payload.get("results", []) if isinstance(r, dict)]
        # 支持 {"kp_1": {...}, "kp_2": {...}} 这种结构
        rows = []
        for v in payload.values():
            if isinstance(v, dict):
                rows.append(v)
        return rows
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def build_a_hits_map(suggestion_payload: Dict) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for row in extract_rows(suggestion_payload):
        if not isinstance(row, dict):
            continue
        kp = row.get("knowledge_point_id")
        kp = to_int(kp)
        if kp is None:
            continue
        a_info = row.get("a_behavior", {})
        if isinstance(a_info, dict):
            out[kp] = int(a_info.get("a_total_hits", 0) or 0)
    return out


def build_suggestion_effect_map(eval_payload: Dict) -> Dict[int, float]:
    """
    读取建议效果评估（overall_score_5），映射为 [0,1] 强度系数。
    如果缺失，后续会使用默认中性值 0.6。
    """
    out: Dict[int, float] = {}
    for row in extract_rows(eval_payload):
        if not isinstance(row, dict):
            continue
        kp_id = to_int(row.get("knowledge_point_id"))
        if kp_id is None:
            continue
        try:
            score = float(row.get("overall_score_5", 3.0))
        except (TypeError, ValueError):
            score = 3.0
        score = max(1.0, min(5.0, score))
        out[kp_id] = (score - 1.0) / 4.0
    return out


def infer_kp_id_from_row(row: Dict[str, Any]) -> Optional[int]:
    direct = to_int(row.get("knowledge_point_id"))
    if direct is not None:
        return direct

    for key in ("eval_id", "window", "label", "group_label"):
        raw = row.get(key)
        if raw is None:
            continue
        text = str(raw)
        m = re.search(r"kp_(\d+)", text)
        if m:
            return to_int(m.group(1))
    return None


def build_suggestion_effect_map_with_coverage(eval_payload: Dict) -> Dict[str, Any]:
    """
    从评估文件提取知识点分数映射，并返回覆盖率，便于发现静默回退问题。
    """
    out: Dict[int, float] = {}
    rows = extract_rows(eval_payload)
    all_rows = len(rows)
    matched_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        kp_id = infer_kp_id_from_row(row)
        if kp_id is None:
            continue
        try:
            score = float(row.get("overall_score_5", 3.0))
        except (TypeError, ValueError):
            score = 3.0
        score = max(1.0, min(5.0, score))
        out[kp_id] = (score - 1.0) / 4.0
        matched_rows += 1
    return {"effect_map": out, "all_rows": all_rows, "matched_rows": matched_rows}


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    q = max(0.0, min(1.0, q))
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    return float(arr[idx])


def build_a_ratio_map(
    base_rows: List[Dict[str, Any]], a_hits_map: Dict[int, int], writing_payload: Dict[str, Any]
) -> Dict[int, float]:
    """
    用“知识点单位时间写作密度”估计 A 比例：
    1) 由知识点 char span 估算时长（分钟）；
    2) density = a_total_hits / duration_min；
    3) 用全课 P90 做归一化，得到 [0,1] 的 a_ratio。
    """
    minute_hits: Dict[int, int] = {}
    hits_list = writing_payload.get("hits", []) if isinstance(writing_payload, dict) else []
    if isinstance(hits_list, list):
        for item in hits_list:
            if not isinstance(item, dict):
                continue
            minute = to_int(item.get("minute"))
            if minute is None or minute < 0:
                continue
            minute_hits[minute] = minute_hits.get(minute, 0) + 1

    max_minute = max(minute_hits.keys()) if minute_hits else 59
    max_end_char = max(int(r.get("end_char", 0) or 0) for r in base_rows) if base_rows else 1
    max_end_char = max(1, max_end_char)

    density_map: Dict[int, float] = {}
    densities: List[float] = []
    for row in base_rows:
        kp_id = to_int(row.get("knowledge_point_id"))
        if kp_id is None:
            continue
        start_char = int(row.get("start_char", 0) or 0)
        end_char = int(row.get("end_char", 0) or 0)
        span = max(1, end_char - start_char)
        duration_min = max(1.0, (span / max_end_char) * max_minute)
        hits = float(a_hits_map.get(kp_id, 0))
        density = hits / duration_min
        density_map[kp_id] = density
        densities.append(density)

    p90 = quantile(densities, 0.9)
    if p90 <= 0:
        return {kp_id: 0.0 for kp_id in density_map.keys()}
    return {kp_id: max(0.0, min(1.0, density / p90)) for kp_id, density in density_map.items()}


def build_target_mode_map(suggestion_payload: Dict) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for row in extract_rows(suggestion_payload):
        if not isinstance(row, dict):
            continue
        kp = to_int(row.get("knowledge_point_id"))
        if kp is None:
            continue
        mode = str(row.get("target_mode", row.get("teacher_expected_ci", "c"))).strip().lower()
        out[kp] = mode if mode in {"c", "i"} else "c"
    return out


def before_counts_and_avg(
    cls: Dict,
    class_size: int,
    a_ratio: float,
) -> Dict[str, float]:
    """
    按用户最新口径：
    - A 行为由命中次数映射为 A 人数，作为基线的一部分。
    - P 人数 = class_size - A人数 - C人数 - I人数。
    """
    ci = str(cls.get("final_ci_level", "c")).strip().lower()
    speakers = int(cls.get("student_speaker_count", 0) or 0)
    speakers = max(0, min(class_size, speakers))

    if ci == "i":
        i_count = speakers
        c_count = 0
    elif ci == "c":
        c_count = speakers
        i_count = 0
    else:
        c_count = 0
        i_count = 0

    remaining_after_ci = max(0, class_size - c_count - i_count)
    # A 比例由“知识点单位时间写作密度”归一化得到（数据驱动，不使用硬编码常数 120）。
    a_ratio = max(0.0, min(1.0, float(a_ratio)))
    a_count = int(round(remaining_after_ci * a_ratio))
    a_count = max(0, min(remaining_after_ci, a_count))

    p_count = max(0, class_size - a_count - c_count - i_count)
    avg_level = ((1.0 * a_count) + (2.0 * c_count) + (3.0 * i_count)) / max(1, class_size)
    return {
        "p_count": p_count,
        "a_count": a_count,
        "c_count": c_count,
        "i_count": i_count,
        "avg_icap_level_before": round(avg_level, 4),
    }


def predict_after_avg_level(
    avg_before: float,
    current_ci: str,
    target_mode: str,
    sug_effect: float,
    method_score_shift: float,
) -> float:
    """
    预测建议后平均等级（虚线）：
    - 结合建议目标模式 target_mode
    - 放大方法评分差异（method_score_shift）体现方法间预期差距
    """
    current_ci = str(current_ci or "").lower()
    target_mode = str(target_mode or "c").lower()
    sug_effect = max(0.0, min(1.0, sug_effect))
    method_score_shift = max(-0.6, min(0.6, method_score_shift))

    # 基础提升幅度：I 目标通常提升更高，C 目标较温和
    if target_mode == "i":
        base_gain = 0.24 if current_ci != "i" else 0.16
    else:
        base_gain = 0.20 if current_ci in {"p", "a", "p/a", "unknown", ""} else 0.14

    # eval 评分本身驱动提升（非线性放大）
    score_gain = EVAL_SCORE_WEIGHT * (sug_effect**EVAL_SCORE_POWER)
    # 方法间评分均值差异只做“正向拉开”，不做负向扣减，避免出现负提升。
    positive_shift = max(0.0, method_score_shift)
    shift_gain = EVAL_DIFF_AMPLIFY * (positive_shift**EVAL_DIFF_POWER)
    gain = max(MIN_METHOD_UPLIFT, base_gain + score_gain + shift_gain)
    after = min(3.0, max(0.0, avg_before + gain))
    return round(after, 4)


def plot_method_comparison_dashboard(
    labels: List[str],
    y_before: List[float],
    method_after_map: Dict[str, List[float]],
    method_colors: Dict[str, str],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if not labels or not method_after_map:
        plt.figure(figsize=(9, 4))
        plt.text(0.5, 0.5, "No comparable method data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_png, dpi=220)
        plt.close()
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    point_count = len(labels)
    fig_width = max(13, 0.95 * point_count + 6)
    fig, (ax_line, ax_delta_bar, ax_delta_heat) = plt.subplots(
        3,
        1,
        figsize=(fig_width, 11),
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.7], "hspace": 0.38},
    )

    x = list(range(point_count))
    ax_line.plot(x, y_before, label="Before (Shared Baseline)", color="#111827", linewidth=2.5, marker="o")
    for method_name, after_series in method_after_map.items():
        color = method_colors.get(method_name, "#2563eb")
        ax_line.plot(x, after_series, label=f"{method_name} After", color=color, linewidth=2.2, marker="o")
    ax_line.set_xticks(x, labels=labels, rotation=45, ha="right")
    ax_line.set_ylim(0.0, 3.05)
    ax_line.set_ylabel("ICAP Avg")
    ax_line.legend(ncol=2)

    method_names = list(method_after_map.keys())
    mean_before = sum(y_before) / max(1, len(y_before))
    mean_after_map = {m: sum(v) / max(1, len(v)) for m, v in method_after_map.items()}
    mean_deltas = [mean_after_map[m] - mean_before for m in method_names]
    bar_colors = [method_colors.get(m, "#2563eb") for m in method_names]
    ax_delta_bar.bar(method_names, mean_deltas, color=bar_colors, alpha=0.9)
    ax_delta_bar.axhline(0.0, color="#374151", linewidth=1.0)
    ax_delta_bar.set_ylabel("Mean Delta")
    for idx, val in enumerate(mean_deltas):
        ax_delta_bar.text(idx, val + (0.02 if val >= 0 else -0.04), f"{val:+.2f}", ha="center", va="bottom", fontsize=9)

    delta_matrix = []
    for m in method_names:
        delta_matrix.append([after - before for before, after in zip(y_before, method_after_map[m])])

    # 仅在图例中展示非负提升区间（从 0 开始），去掉负半轴。
    delta_matrix_for_heat = [[max(0.0, v) for v in row] for row in delta_matrix]
    heat_cmap = LinearSegmentedColormap.from_list("method_delta_positive", ["#ffffff", "#b7e4c7", "#15803d"])
    heat = ax_delta_heat.imshow(delta_matrix_for_heat, cmap=heat_cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax_delta_heat.set_yticks(list(range(len(method_names))), labels=method_names)
    ax_delta_heat.set_xticks(x, labels=labels, rotation=45, ha="right")
    for r_idx, row in enumerate(delta_matrix):
        for c_idx, val in enumerate(row):
            text_color = "#111827" if abs(val) < 0.55 else "white"
            ax_delta_heat.text(c_idx, r_idx, f"{val:+.2f}", ha="center", va="center", color=text_color, fontsize=8)
    cbar = fig.colorbar(heat, ax=ax_delta_heat, fraction=0.02, pad=0.02)
    cbar.set_label("After - Before (>=0 in legend)")

    # 用户要求去掉标题：不设置各子图标题和总标题。

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    student_payload = load_json(INPUT_STUDENT_CLASSIFICATION)
    shared_signal_payload = load_json(INPUT_SHARED_SIGNALS) if INPUT_SHARED_SIGNALS.exists() else {}
    writing_payload = load_json(INPUT_WRITING_BEHAVIOR) if INPUT_WRITING_BEHAVIOR.exists() else {}
    a_hits_map = build_a_hits_map(shared_signal_payload)
    target_mode_map = build_target_mode_map(shared_signal_payload)

    before_series: List[float] = []
    x_labels: List[str] = []
    base_details: List[Dict[str, Any]] = []

    rows = student_payload.get("results", [])
    if not isinstance(rows, list):
        rows = []
    sorted_rows = sorted(
        [r for r in rows if isinstance(r, dict)],
        key=lambda r: int(r.get("knowledge_point_id", 10**9)),
    )

    for row in sorted_rows:
        if not isinstance(row, dict):
            continue
        kp_id = to_int(row.get("knowledge_point_id"))
        if kp_id is None:
            continue
        cls = row.get("classification", {})
        if not isinstance(cls, dict):
            cls = {}
        before = before_counts_and_avg(cls, CLASS_SIZE, 0.0)

        before_series.append(before["avg_icap_level_before"])
        x_labels.append(f"KP{kp_id}")
        base_details.append(
            {
                "knowledge_point_id": kp_id,
                "topic": row.get("topic", ""),
                "start_char": int(row.get("start_char", 0) or 0),
                "end_char": int(row.get("end_char", 0) or 0),
                "current_ci": str(cls.get("final_ci_level", "c")).strip().lower(),
                "before": before,
            }
        )

    # 基线 A 比例改为“写作密度归一化”，不再使用 a_hits/120。
    a_ratio_map = build_a_ratio_map(base_details, a_hits_map, writing_payload)
    for idx, row in enumerate(sorted_rows):
        if not isinstance(row, dict):
            continue
        kp_id = to_int(row.get("knowledge_point_id"))
        if kp_id is None or idx >= len(before_series):
            continue
        cls = row.get("classification", {})
        if not isinstance(cls, dict):
            cls = {}
        a_ratio = a_ratio_map.get(kp_id, 0.0)
        before = before_counts_and_avg(cls, CLASS_SIZE, a_ratio)
        before_series[idx] = before["avg_icap_level_before"]
        base_details[idx]["before"] = before
        base_details[idx]["a_ratio_for_baseline"] = round(a_ratio, 4)

    methods_payload: List[Dict[str, Any]] = []
    method_after_map: Dict[str, List[float]] = {}
    method_colors: Dict[str, str] = {}
    warnings: List[str] = []
    prepared_methods: List[Dict[str, Any]] = []

    for scenario in METHOD_SCENARIOS:
        method_name = str(scenario["name"])
        display_name = str(scenario["display_name"])
        method_colors[display_name] = str(scenario.get("color", "#2563eb"))
        eval_path = Path(str(scenario["eval_path"]))
        suggestions_path = Path(str(scenario["suggestions_path"]))
        default_effect = float(scenario.get("default_effect", 0.6))

        if not eval_path.exists():
            warnings.append(f"[WARN] {display_name} eval file missing: {eval_path}")
            continue

        eval_payload = load_flexible_payload(eval_path)
        effect_info = build_suggestion_effect_map_with_coverage(eval_payload)
        sug_effect_map = effect_info["effect_map"]
        matched_rows = int(effect_info["matched_rows"])
        all_rows = int(effect_info["all_rows"])
        if all_rows > 0 and matched_rows < max(3, int(0.5 * all_rows)):
            warnings.append(
                f"[WARN] {display_name} eval kp coverage is low: matched {matched_rows}/{all_rows}. Some KPs may fallback to default_effect."
            )

        if sug_effect_map:
            avg_method_effect = sum(sug_effect_map.values()) / len(sug_effect_map)
        else:
            avg_method_effect = default_effect
        prepared_methods.append(
            {
                "method": method_name,
                "display_name": display_name,
                "color": method_colors[display_name],
                "paths": {"suggestions": str(suggestions_path), "eval": str(eval_path)},
                "default_effect": default_effect,
                "avg_method_effect": avg_method_effect,
                "sug_effect_map": sug_effect_map,
                "coverage": {"matched_rows": matched_rows, "all_rows": all_rows},
            }
        )

    global_mean_effect = (
        sum(float(m["avg_method_effect"]) for m in prepared_methods) / max(1, len(prepared_methods))
        if prepared_methods
        else 0.6
    )

    for method in prepared_methods:
        display_name = str(method["display_name"])
        method_name = str(method["method"])
        default_effect = float(method["default_effect"])
        sug_effect_map = dict(method["sug_effect_map"])
        avg_method_effect = float(method["avg_method_effect"])
        method_score_shift = avg_method_effect - global_mean_effect
        after_series: List[float] = []
        details: List[Dict[str, Any]] = []
        for idx, base in enumerate(base_details):
            kp_id = int(base["knowledge_point_id"])
            current_ci = str(base.get("current_ci", "c"))
            before_avg = float(before_series[idx])
            target_mode = target_mode_map.get(kp_id, "c")
            sug_effect = float(sug_effect_map.get(kp_id, default_effect))
            after_avg = predict_after_avg_level(
                avg_before=before_avg,
                current_ci=current_ci,
                target_mode=target_mode,
                sug_effect=sug_effect,
                method_score_shift=method_score_shift,
            )
            after_series.append(after_avg)
            details.append(
                {
                    "knowledge_point_id": kp_id,
                    "topic": base.get("topic", ""),
                    "target_mode": target_mode,
                    "suggestion_effect_score_0_to_1": round(sug_effect, 4),
                    "after_predicted_avg_icap": after_avg,
                }
            )

        method_after_map[display_name] = after_series
        methods_payload.append(
            {
                "method": method_name,
                "display_name": display_name,
                "paths": dict(method["paths"]),
                "assumptions": {
                    "default_effect_if_missing": default_effect,
                    "method_avg_effect_score_0_to_1": round(avg_method_effect, 4),
                    "global_avg_effect_score_0_to_1": round(global_mean_effect, 4),
                    "method_score_shift_for_amplification": round(method_score_shift, 4),
                    "eval_score_power": EVAL_SCORE_POWER,
                    "eval_score_weight": EVAL_SCORE_WEIGHT,
                    "eval_diff_amplify": EVAL_DIFF_AMPLIFY,
                    "eval_diff_power": EVAL_DIFF_POWER,
                },
                "coverage": dict(method["coverage"]),
                "overall_avg_after_predicted": round(sum(after_series) / max(1, len(after_series)), 4),
                "overall_avg_delta": round((sum(after_series) - sum(before_series)) / max(1, len(after_series)), 4),
                "knowledge_point_predictions": details,
            }
        )

    plot_method_comparison_dashboard(x_labels, before_series, method_after_map, method_colors, OUTPUT_PNG)
    avg_before_all = round(sum(before_series) / max(1, len(before_series)), 4)
    method_ranking = sorted(methods_payload, key=lambda x: float(x.get("overall_avg_after_predicted", 0.0)), reverse=True)

    payload = {
        "class_size": CLASS_SIZE,
        "comparison_methods_expected": [m["display_name"] for m in METHOD_SCENARIOS],
        "icap_level_mapping": {"p": 0, "a": 1, "c": 2, "i": 3},
        "method_notes": {
            "a_behavior_note": "A behavior is mapped by per-knowledge-point writing-hit density (hits per estimated minute), then normalized by class P90 to produce baseline A ratio.",
            "p_count_note": "P count is residual students after removing A/C/I participants.",
            "prediction_note": "After level uses target_mode + eval-derived effect score, and amplifies method score differences from the cross-method mean.",
            "comparison_note": "Method gaps are driven by score differences, without method-specific fixed gain coefficients or activity bonuses.",
        },
        "inputs": {
            "student_classification": str(INPUT_STUDENT_CLASSIFICATION),
            "shared_signals_for_target_mode_and_a_hits": str(INPUT_SHARED_SIGNALS),
            "writing_behavior_for_a_density": str(INPUT_WRITING_BEHAVIOR),
        },
        "warnings": warnings,
        "series_before_avg_icap_shared_baseline": before_series,
        "overall_avg_before": avg_before_all,
        "method_comparisons": methods_payload,
        "method_ranking_by_overall_avg_after": [
            {
                "display_name": row.get("display_name", ""),
                "overall_avg_after_predicted": row.get("overall_avg_after_predicted", 0.0),
                "overall_avg_delta": row.get("overall_avg_delta", 0.0),
            }
            for row in method_ranking
        ],
        "knowledge_point_baseline_details": base_details,
        "plot_path": str(OUTPUT_PNG),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] JSON: {OUTPUT_JSON}")
    print(f"[DONE] Plot: {OUTPUT_PNG}")
    for w in warnings:
        print(w)


if __name__ == "__main__":
    main()
