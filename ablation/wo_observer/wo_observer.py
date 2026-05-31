# gpt-4o:[Efficiency] time=15.151s, tokens=23501 (prompt=19304, completion=4197), est_cost=$0.159475, throughput=39.601 kp/min
#gpt-4.1-mini :[Efficiency] time=15.716s, tokens=22102 (prompt=16567, completion=5535), est_cost=$0.16586, throughput=41.995 kp/min
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------- 配置与路径 ----------
PATHS = {
    "knowledge": Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Meeting for 3D Design/icap_output/icap_split_free/knowledge_segments_global_free.json"),
    "teacher_curr": Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Meeting for 3D Design/icap_output/icap_teacher/teacher_current_ci_by_knowledge.json"),
    "teacher_exp": Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Meeting for 3D Design/icap_output/icap_exp/teacher_expected_ci_by_knowledge.json"),
    "out_dir": Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm/ablation/wo_observer/stem")
}

MODEL = "openai/gpt-4.1-mini"
TIMEOUT = 90
MAX_WORKERS = 20
DEFAULT_PROMPT_COST_PER_1M = 3.0
DEFAULT_COMPLETION_COST_PER_1M = 15.0


def ci_rank(level: str) -> int:
    order = {"p": 0, "a": 1, "c": 2, "i": 3}
    return order.get(str(level or "").strip().lower(), 0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ICAP teaching suggestions.")
    parser.add_argument("--class-size", type=int, default=30, help="Total enrolled participants.")
    args = parser.parse_args()
    if args.class_size <= 0:
        raise ValueError("Class size must be positive.")
    return args

def build_client() -> OpenAI:
    if not OpenAI:
        raise ImportError("Please install openai: pip install openai")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API Key in environment variables.")
    
    custom_base_url = os.getenv("LLM_BASE_URL", "").strip()
    if custom_base_url:
        base_url = custom_base_url
    else:
        base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
        
    return OpenAI(api_key=api_key, base_url=base_url, timeout=TIMEOUT)

# ---------- 通用数据加载器 (核心去重) ----------
def load_json_records(path: Path, key_name: str) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(key_name, [])

def build_unified_kp_map() -> Dict[int, Dict[str, Any]]:
    """合并来自5个不同底层JSON的数据"""
    kp_map = {}
    
    # 1. 基础知识点段落
    try:
        for seg in load_json_records(PATHS["knowledge"], "segments"):
            kp_id = seg.get("knowledge_point_id")
            if isinstance(kp_id, int):
                kp_map[kp_id] = {
                    "topic": str(seg.get("topic", "")).strip(),
                    "summary": str(seg.get("summary", "")).strip(),
                    "text": str(seg.get("text", "")).strip(),
                    "start_char": int(seg.get("start_char", 0)),
                    "end_char": int(seg.get("end_char", 0)),
                }
    except FileNotFoundError:
        print(f"Warning: Base knowledge file not found. Creating empty map.")
            
    # 2. 教师当前与预期 CI 状态
    try:
        for row in load_json_records(PATHS["teacher_curr"], "results"):
            kp_id = row.get("knowledge_point_id")
            if kp_id in kp_map:
                kp_map[kp_id].update({
                    "teacher_current_possible_ci": str(row.get("expected_ci_level", "c")).lower(),
                    "teacher_current_reason": str(row.get("expected_reason", "")),
                })
    except FileNotFoundError:
        pass
            
    try:
        for row in load_json_records(PATHS["teacher_exp"], "results"):
            kp_id = row.get("knowledge_point_id")
            if kp_id in kp_map:
                kp_map[kp_id].update({
                    "teacher_expected_ci": str(row.get("expected_ci_level", "c")).lower(),
                    "teacher_expected_reason": str(row.get("expected_reason", "")),
                })
    except FileNotFoundError:
        pass
            
    return kp_map

# ---------- 核心 LLM 驱动与 Prompt 架构升级 ----------
def get_system_prompt() -> str:
    # 优化点：1. 严厉禁止 C 状态出现任何交互词汇； 2. 显式要求发挥想象力，输出多样化、有创意的支架策略。
    return """You are an expert teaching coach analyzing classroom interaction using ICAP theory.
Your response MUST be a single, valid JSON object matching the requested schema. Do not output markdown code blocks.

[OUTPUT DIVERSITY]
- Fully leverage your pedagogical knowledge to provide highly diverse, non-repetitive, and innovative scaffolding strategies. Avoid generic templates.

[ICAP Ladder Constraints]
- Progression: P -> A -> C -> I. 
- CRITICAL FOR TARGET MODE = C (Constructive): 
  * The goal is strictly INDIVIDUAL cognitive construction. Participants must generate knowledge/outputs entirely on their own (e.g., individual drawing, solo mapping, independent reflection, teacher probing a single participant).
  * ABSOLUTELY FORBIDDEN concepts, behaviors, or words: 'group', 'peer', 'partner', 'share with others', 'discuss', 'collaborate', 'team', 'talk to neighbor'. Any form of peer interaction is a SEVERE failure for Mode C.
- CRITICAL FOR TARGET MODE = I (Interactive): 
  * The strategy MUST explicitly facilitate peer-to-peer interaction, debate, collaborative problem-solving, or partner dialogue. Purely solo tasks are strictly insufficient."""

def get_user_prompt(kp_id: int, data: Dict, class_size: int) -> str:
    t_ci = data.get("teacher_current_possible_ci", "c")
    e_ci = data.get("teacher_expected_ci", "c")
    
    stage_rank = {"p": 0, "a": 1, "c": 2, "i": 3}
    distance = max(0, stage_rank.get(e_ci, 2) - stage_rank.get(t_ci, 1))
    level_achieved = ci_rank(t_ci) >= ci_rank(e_ci)

    parts = [p.strip() for p in data.get("text", "").split(".") if p.strip()]
    snippets = [p[:200] for p in parts if "?" in p or "what" in p.lower()][:3] or parts[:2]

    return f"""Analyze the following Knowledge Point Context:
- ID: {kp_id} | Topic: {data.get('topic')}
- Current Teaching CI Signal: {t_ci} | Target Expected CI: {e_ci}
- Metrics: Class Size={class_size}
- Transcript Snippets: {json.dumps(snippets, ensure_ascii=False)}
- Level achieved flag (trusted): {str(level_achieved).lower()}
- Stage distance (trusted): {distance}

[Transcript Full Excerpt]
{data.get('text')}

[MANDATORY TWO-STEP DIAGNOSIS]
You must strictly execute these two steps in order:
Step 1 (Level):
- Evaluate whether the currently observed cognitive mode reaches the target mode.
- Output this in gap_diagnosis explicitly.
Step 2 (Breadth):
- Evaluate participation breadth and whole-class coverage.
- Explain whether only a small visible subset reaches the target while most participants remain passive.

[HARD POLICY]
- If Step 1 says Level is already achieved, DO NOT propose a "restart" or "mode overthrow".
- In that case, proposed_rewrite and scaffolding_strategy MUST focus on:
  "how to scale this already-achieved cognitive mode to silent participants."
- This means replication, diffusion, rotation, and broader participation expansion, not redefining the core mode.

Output Requirements:
Return a JSON object exactly containing these keys:
{{
  "knowledge_point_id": {kp_id},
  "target_mode": "{e_ci}",
  "gap_diagnosis": "Explain why coverage is bottlenecked and if behavior converted into outputs. Quote transcript.",
  "proposed_rewrite": "Concept-level pedagogical modification to transition into target mode.",
  "scaffolding_strategy": "1-2 highly operational strategies. Align tightly with the specific constraints for Mode C or Mode I specified in system instructions.",
  "new_question": "One targeted teacher question forcing the exact target_mode behavior.",
  "reference_answer": "High-quality sample response.",
}}"""

def get_pricing_config() -> Dict[str, float]:
    # 1. 定义一个常用模型的价格字典 (每 1M tokens 的美金价格)
    MODEL_PRICING_MAP = {
        "gpt-4o": {"prompt": 5.0, "completion": 15.0},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "anthropic/claude-3.5-sonnet": {"prompt": 3.0, "completion": 15.0},
        "gpt-5-mini": {"prompt": 0.30, "completion": 1.20}, # 举例：请根据 2026 年实际市价修改
        # 你可以继续添加 openrouter 的模型，例如 "anthropic/claude-3.5-sonnet"
    }
    
    # 2. 根据当前的全局变量 MODEL 获取对应的基础价格，如果找不到则用默认值
    model_fee = MODEL_PRICING_MAP.get(MODEL, {"prompt": DEFAULT_PROMPT_COST_PER_1M, "completion": DEFAULT_COMPLETION_COST_PER_1M})
    
    # 3. 依然保留环境变量的高优先级，方便外部临时覆盖
    prompt = float(os.getenv("LLM_PROMPT_COST_PER_1M", str(model_fee["prompt"])))
    completion = float(os.getenv("LLM_COMPLETION_COST_PER_1M", str(model_fee["completion"])))
    
    return {
        "prompt_cost_per_1m": max(0.0, prompt),
        "completion_cost_per_1m": max(0.0, completion),
    }

def compute_cost_usd(prompt_tokens: int, completion_tokens: int, pricing: Dict[str, float]) -> float:
    prompt_cost = (max(0, prompt_tokens) / 1_000_000.0) * pricing["prompt_cost_per_1m"]
    completion_cost = (max(0, completion_tokens) / 1_000_000.0) * pricing["completion_cost_per_1m"]
    return prompt_cost + completion_cost


def safe_json_loads(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if not content:
        raise ValueError("Empty model response content.")
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model response: {content[:300]}")
        return json.loads(match.group(0))


def call_llm(client: OpenAI, sys_prompt: str, user_prompt: str, pricing: Dict[str, float]) -> Dict:
    started = time.perf_counter()
    try:
        current_model = MODEL
        if "openrouter.ai" in str(client.base_url) and current_model == "gpt-4o-mini":
            current_model = "openai/gpt-4o-mini"

        request_kwargs: Dict[str, Any] = {
            "model": current_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.75,
        }
        # json_object is not consistently supported by non-OpenAI families (e.g., Claude via routers).
        if "claude" not in current_model.lower():
            request_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**request_kwargs)
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", (prompt_tokens + completion_tokens)) or 0)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        cost_usd = compute_cost_usd(prompt_tokens, completion_tokens, pricing)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        payload = safe_json_loads(response.choices[0].message.content or "")
        return {
            "payload": payload,
            "metrics": {
                "elapsed_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(cost_usd, 6),
            },
        }
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "payload": {"error": f"LLM failed: {str(e)}", "confidence": 0.0},
            "metrics": {
                "elapsed_ms": elapsed_ms,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        }

# ---------- 流程控制与多线程 ----------
def process_kp(client: OpenAI, kp_id: int, data: Dict, class_size: int, pricing: Dict[str, float]) -> Dict:
    sys_prompt = get_system_prompt()
    user_prompt = get_user_prompt(kp_id, data, class_size)
    
    llm_result = call_llm(client, sys_prompt, user_prompt, pricing)
    res = llm_result.get("payload", {})
    
    meta_info = {
        "knowledge_point_id": kp_id,
        "topic": data.get("topic"),
        "teacher_current_ci": data.get("teacher_current_possible_ci", "c"),
        "teacher_expected_ci": data.get("teacher_expected_ci", "c"),
    }
    merged = {**meta_info, **res}
    merged["efficiency"] = llm_result.get("metrics", {})
    return merged

def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    PATHS["out_dir"].mkdir(parents=True, exist_ok=True)
    pricing = get_pricing_config()
    
    print("[1/4] Loading and merging data from JSON files...")
    kp_map = build_unified_kp_map()
    
    print(f"[2/3] Requesting LLM suggestions across {len(kp_map)} KPs using ThreadPool...")
    client = build_client()
    final_results = []
    
    if not kp_map:
        print("No knowledge points found to process.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_kp = {
            executor.submit(process_kp, client, kp_id, data, args.class_size, pricing): kp_id
            for kp_id, data in kp_map.items()
        }
        
        for future in as_completed(future_to_kp):
            kp_id = future_to_kp[future]
            try:
                item = future.result()
                final_results.append(item)
                print(f" -> [SUCCESS] Knowledge Point {kp_id} processed.")
            except Exception as e:
                print(f" -> [FAILURE] Knowledge Point {kp_id} raised exception: {e}")

    final_results.sort(key=lambda x: x["knowledge_point_id"])

    total_elapsed_ms = int((time.perf_counter() - t0) * 1000)
    total_prompt_tokens = sum(int(item.get("efficiency", {}).get("prompt_tokens", 0)) for item in final_results)
    total_completion_tokens = sum(int(item.get("efficiency", {}).get("completion_tokens", 0)) for item in final_results)
    total_tokens = sum(int(item.get("efficiency", {}).get("total_tokens", 0)) for item in final_results)
    total_cost_usd = round(sum(float(item.get("efficiency", {}).get("estimated_cost_usd", 0.0)) for item in final_results), 6)
    avg_kp_elapsed_ms = int(
        sum(int(item.get("efficiency", {}).get("elapsed_ms", 0)) for item in final_results) / max(1, len(final_results))
    )
    throughput_kp_per_min = round((len(final_results) / max(1e-6, total_elapsed_ms / 1000.0)) * 60.0, 3)
    efficiency_summary = {
        "total_elapsed_ms": total_elapsed_ms,
        "total_elapsed_sec": round(total_elapsed_ms / 1000.0, 3),
        "avg_kp_elapsed_ms": avg_kp_elapsed_ms,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "estimated_total_cost_usd": total_cost_usd,
        "throughput_kp_per_min": throughput_kp_per_min,
        "pricing": {
            "prompt_cost_per_1m": pricing["prompt_cost_per_1m"],
            "completion_cost_per_1m": pricing["completion_cost_per_1m"],
        },
        "pricing_note": "Estimated cost based on token usage and configurable per-1M-token pricing.",
    }

    print("[3/3] Writing output payloads...")
    output_json = PATHS["out_dir"] / "icap_suggestions.json"
    output_json.write_text(
        json.dumps(
            {
                "model": MODEL,
                "class_size": args.class_size,
                "efficiency_summary": efficiency_summary,
                "results": final_results
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )
    print(
        "[Efficiency] "
        f"time={efficiency_summary['total_elapsed_sec']}s, "
        f"tokens={efficiency_summary['total_tokens']} "
        f"(prompt={efficiency_summary['total_prompt_tokens']}, completion={efficiency_summary['total_completion_tokens']}), "
        f"est_cost=${efficiency_summary['estimated_total_cost_usd']}, "
        f"throughput={efficiency_summary['throughput_kp_per_min']} kp/min"
    )
    print(f"Perfect process finished. Results stored in: {output_json}")

if __name__ == "__main__":
    main()