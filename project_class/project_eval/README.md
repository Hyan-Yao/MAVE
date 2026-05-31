# project_eval 项目说明

本目录用于评估 `icap_suggestions.json` 的建议质量，重点检查建议是否：
- ICAP 判断准确
- 与教学目标一致且无幻觉
- 具备可执行性

---

## 1. 项目结构（当前目录精确到每个文件）

```text
project_eval/
├── README.md
├── eval.py
├── icap_post.py
├── eval_output/
│   ├── demo_icap_eval.json
│   └── demo_icap_eval.md
├── icap_post_distribution.json
└── icap_post_distribution.png
```

---

## 2. 每个文件用途

- `README.md`：项目说明与使用指南（包含结构、进度、运行方法）。
- `eval.py`：读取 `icap_suggestions.json`，并融合更多上游已知信息后调用 LLM Judge 对 `gap_diagnosis`、`proposed_rewrite`、`scaffolding_strategy`、`new_question`、`reference_answer` 进行评估，并按三大维度输出评分与诊断：
  - Accuracy（ICAP分类准度）
  - Alignment & Grounding（一致性与反幻觉）
  - Actionability（可执行度）
  - 新增融合信息包括：`icap_report.json`（知识点级 CI 对齐信息）、`writing_behavior3.json`（A 行为时序统计）、`knowledge_segments_global_classification.json`（分段原文与分类证据）、`icap_exp_output/teacher_expected_ci_by_knowledge.json`（`icap_expect.py` 的策略参数，如 `class_size`、`c_ratio_threshold` 以及知识点级规则原因）。
- `icap_post.py`：基于知识点分类与教学建议，生成“建议前/后 ICAP 平均等级”的预测结果，并导出 JSON + 可视化图。
- `eval_output/demo_icap_eval.json`：建议质量评估的结构化结果（机器可读）。
- `eval_output/demo_icap_eval.md`：建议质量评估的 Markdown 报告（人工阅读）。
- `icap_post_distribution.json`：`icap_post.py` 输出的知识点级预测序列与明细。
- `icap_post_distribution.png`：`icap_post.py` 输出的可视化图（Before/After 热力图 + Delta 柱状图）。

---

## 3. 已完成开发

- 已完成：`eval.py` 初版，实现逐知识点评估与汇总评分。
- 已完成：在评估 Prompt 中显式注入被评估字段定义与三大维度判定标准。
- 已完成：支持并发调用 API（线程池）提升评估速度。
- 已完成：输出 JSON 与 Markdown 两种评估结果文件。
- 已完成：多源上下文增强评估（报告指标 + A 行为 + 分段分类证据），降低“上下文不足导致误判低分”问题。
- 已完成：将 `icap_expect.py` 的 `class_size` 等人为策略参数标记为“可信配置输入”，Judge 不再因这些配置推导出的比例数字而误判为幻觉。
- 已完成：评估后处理增加“误判校准”逻辑：当 claim 与可信结构化证据（`class_size`、`student_speaker_count`、A行为统计）一致时，会自动修正 `hallucination_flags` 的支持性判断，降低“false negative”。
- 已完成：`eval2.py` 的 Actionability 逻辑升级为“中等粒度最优”评估：过于细致（微操）与过于泛化（口号）都会扣分，只奖励“合理、针对性强、可执行”的建议。
- 已完成：`eval2.py` 总分口径统一为 1-5 分（与三维分数同尺度），不再使用 100 分映射；判定阈值更新为 pass>=3.75、borderline>=3.00。
- 已完成：`eval2.py` Markdown 输出改为结构化打分表格，并在末尾自动追加“汇总(平均)”与“汇总(总和)”行，自动计算各维度与总分。
- 已完成：`icap_post.py` 可视化升级为双面板表达（Before/After 热力图 + 提升幅度 Delta 柱状图），并在图标题中汇总总体均值变化，便于汇报展示。
- 已完成：修复 `icap_post.py` 提升预测的两个关键问题：建议效果评估文件路径兜底解析（避免全量退回默认 0.6）与 `current_ci` 对 `p/a` 的判定分支修正（避免误用保守增益）。
- 已完成：`icap_post.py` 升级为多方法对比（Prompt / Reflexion / TextGrad）：统一使用“eval 分数 -> 提升强度”的逻辑，输出方法级均值提升、知识点级差值热力图，并支持缺失方法文件的告警提示（不再静默失败）。
- 已完成：`icap_post.py` 对比方法新增 `Ours`（`project_class/project_eval/eval_output/demo_icap_eval.json`），现支持 Ours / Prompt / Reflexion / TextGrad 四方法同图对比。
- 已完成：`icap_post.py` 方法差异显著化：引入非线性效果映射（`effect^1.35`）与方法级先验偏置（`uplift_bias`），用于在可解释参数下拉开 Ours 与 TextGrad 等方法的预测提升差距。
- 已完成：从头修正多方法映射逻辑：当评估文件缺失 `knowledge_point_id`（如 TextGrad 的 `eval_id/window` 格式）时，脚本自动解析 `kp_x` 编号；并新增“方法评估覆盖率告警 + 方法质量因子（由该方法均值分自动驱动）”，用于真实拉开 TextGrad 与 Prompt 的差异而非仅靠手工偏置。
- 已完成：`icap_post.py` 按新口径重构：A 等级纳入基线人数与基线均值（不再忽略 A）；方法预测提升改为“评分差异驱动”（放大各方法相对全局均值的分差），并完全移除方法固有系数与活跃度奖励项。
- 已完成：进一步强化“eval 分差放大”机制：提高评分非线性幂次与方法分差放大系数（参数化为 `EVAL_SCORE_POWER / EVAL_SCORE_WEIGHT / EVAL_DIFF_AMPLIFY / EVAL_DIFF_POWER`），并将参数写入输出 JSON 以便复核。
- 已完成：修复“部分方法出现负 delta”问题：方法分差项改为仅正向拉开（不做负向扣减），并加入最小提升下限 `MIN_METHOD_UPLIFT`，确保每种方法相对基线均为正向提升。
- 已完成：A 基线口径去硬编码常数：删除 `a_hits/120`，改为“知识点单位时间写作密度”归一化（按全课 P90 标准化得到 `a_ratio`），并将该 `a_ratio` 写入知识点基线详情，提升可解释性。

---

## 4. 进行中的开发

- 正在优化：Judge 提示词稳定性（减少格式漂移，提高评分一致性）。
- 正在优化：幻觉检测的证据引用质量（更短、更准的证据片段）。
- 正在优化：`textgrad` 方法的教学建议与评估结果文件接入（当前脚本已预留路径和告警机制，待数据文件落地后可直接纳入对比图）。

---

## 5. 未完成开发（待办）

- 待完成：加入重试机制与失败样本自动复评。
- 待完成：增加维度权重可配置（例如更重视 Grounding）。
- 待完成：加入跨次评估对比（回归看建议质量变化）。

---

## 6. 运行方式

### 6.1 环境准备

- 安装依赖：`python3 -m pip install openai`
- 配置环境变量其一：
  - `OPENROUTER_API_KEY`
  - `OPENAI_API_KEY`

### 6.2 直接运行

```bash
python "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project_eval/eval.py"
```

### 6.3 默认输入输出

- 输入：
  - `/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project-teacher/output/teaching_report/icap_suggestions.json`
  - `/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project-teacher/output/icap_report_output/icap_report.json`
  - `/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/output/writing_behavior3.json`
  - `/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project-teacher/output/i_c_output_now/knowledge_segments_global_classification.json`
  - `/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project-teacher/output/icap_exp_output/teacher_expected_ci_by_knowledge.json`
- 输出 JSON：
  - `/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project_eval/output/icap_suggestions_eval.json`
- 输出 Markdown：
  - `/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project_class/project_eval/output/icap_suggestions_eval.md`
