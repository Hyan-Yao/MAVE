# MAVE: A Multi-Agent Framework for Grounded Pedagogical Feedback from Online Class Videos
This repository contains the implementation of **MAVE**, a collaborative multi-agent framework designed to transform raw multimodal classroom data into structured, 
pedagogically actionable optimization signals.
## Overview
Timely pedagogical feedback is critical for effective online instruction, yet existing AI-based educational assessment systems still struggle to generate actionable and trustworthy recommendations from authentic classroom environments. Major challenges include long-context noisy videos and weak grounding in real student behaviors. To address these challenges, we introduce **MAVE** (**M**ultimodal **A**gents for **V**ideo-based **E**ducational Feedback), a collaborative multi-agent framework for generating pedagogically grounded feedback from online classroom videos. Built upon the ICAP framework, MAVE formulates feedback generation as a discrepancy reasoning problem between instructors’ intended engagement objectives and students’ observed cognitive states. MAVE decomposes classroom understanding into specialized agents for instructional structuring, multimodal evidence extraction, engagement assessment, and pedagogical recommendation generation. By grounding feedback in temporally aligned multimodal classroom evidence, MAVE produces interpretable and actionable instructional recommendations. Experiments on real-world online classroom videos demonstrate that MAVE generates more pedagogically grounded and actionable feedback than existing LLM-based feedback generation baselines. 

---
## Framework Architecture
<img width="1422" height="623" alt="截屏2026-05-31 18 15 56" src="https://github.com/user-attachments/assets/ee29f3f5-6e83-48ee-862e-d74e00a29948" />

> **Figure**: Overview of MAVE.
> Given a raw online classroom video,
> * **the Curator** segments the lecture and infers the intended ICAP objective for each instructional unit.
> * **The Observer** extracts multimodal student evidence from the corresponding temporal window.
> * **The Assessor** estimates the observed student ICAP state through calibrated ordinal reasoning.
> * **The Advisor** compares the intended and observed ICAP states and generates time-aligned pedagogical feedback grounded in classroom evidence.

## Key Features

Unlike conventional approaches that output flat engagement scores, MAVE treats feedback generation as a **discrepancy reasoning problem** rooted in cognitive psychology.

### 🧠 1. ICAP-Grounded Discrepancy Reasoning
* **Intent vs. Reality Alignment:** Formulates instructional analysis by explicitly evaluating the gap ($\Delta_k$) between an instructor's intended pedagogical objectives and the students' actual observed cognitive states.
* **Psychological Framework:** Fully structures its assessment vocabulary around the standardized **ICAP framework** (Passive, Active, Constructive, Interactive), transforming abstract video feeds into theoretically-grounded dimensions.

### 🤖 2. Collaborative Multi-Agent Architecture
Decouples complex classroom understanding into four highly specialized, coordinated language/vision-language agents to ensure rigorous evaluation:
* **Curator (Pedagogical Structuring):** Automatically parses long lectures into timestamp-aligned, self-contained instructional units via ASR transcripts and infers the ideal target ICAP tier.
* **Observer (Learner Evidence Extraction):** Decomposes video layouts into individual student sub-windows and translates heterogeneous multimodal behavior (gaze, posture, writing device usage, utterances) into dense natural-language summaries.
* **Assessor (Cognitive State Estimation):** Evaluates students' true engagement trajectories from the Observer's summaries using calibrated pedagogical reasoning.
* **Advisor (Feedback Generation):** Compares the target and observed states to synthesize localized, evidence-backed teaching interventions without altering core video data.

### 📊 3. Ordinal-Consistent Confidence Calibration
* **Noise Reduction:** Includes a built-in mathematical calibration layer that enforces ordinal consistency over raw LLM judge boundaries.
* **Robust Predictions:** Penalizes volatile predictions and favors smooth semantic distributions, significantly dampening the local noise and alignment hallucinations typical of generic LLM baselines.

### 🚀 4. Actionable & Grounded Recommendations
* **Context-Local Interventions:** Moves beyond vague suggestions like *"increase interaction"* to provide step-by-step instructional rewrites, concrete scaffolding strategies (e.g., solo visual mapping, reflection journals), and targeted questions.
* **Extensively Validated:** Outperforms common LLM reasoning baselines (*Vanilla VLM*, *Reflexion*, and *TextGrad*) across both STEM and Humanities domains based on extensive expert human and LLM-as-a-Judge evaluations.
* **Backbone Agnostic:** Demonstrates high architectural robustness and stable performance across multiple underlying LLMs, including GPT-4o-mini, Claude 4.5 Sonnet, Gemini 3.5 Flash, and Llama 3.3 70B.

### *there's a demo of our project:*
<img width="1349" height="676" alt="截屏2026-06-02 17 10 16" src="https://github.com/user-attachments/assets/a2859cde-ca1e-41d3-be60-7614cd8c9812" />

## 1. Prerequisites
### 1.1 Environment
Clone the repository and create the environment:
```bash
git clone https://github.com/Hyan-Yao/MAVE.git
cd MAVE
conda env create -f environment.yml
conda activate mave
````
### 1.2 API Key
The scripts read either of the following environment variables by default:
```bash
OPENROUTER_API_KEY

OPENAI_API_KEY
````
Example:
```bash
export OPENROUTER_API_KEY="your_api_key_here"
````
## 2. Data Preparation (Video -> Text)
You need to prepare the course video first and obtain the transcription text. Either of the following methods works:

* **Method A**: Run your own Whisper transcription pipeline (the project already contains relevant video processing and behavior extraction scripts).

* **Method B**: Use YouTube's built-in speech-to-text and export the text.

Ultimately, it is recommended to prepare at least one of the following key inputs:

knowledge_segments_global*.json (Knowledge point segmentation results, which serve as the core input for run_pipeline.py)

writing_behavior.json (Used for Action A-related features)

## 3. One-Click Suggestion Generation (sug)
Main entry script:

/project_class/run_pipeline.py

Most common execution method:

```bash
python "/project_class/run_pipeline.py" \
  --segments-json "your/knowledge_segments_global.json" \
  --writing-behavior-json "your/writing_behavior.json" \
  --output-dir "your_output_directory" \
  --model "your llm"
````
Upon completion, the core suggestion file generated is:

output_directory/icap_suggestions.json

You can also choose to re-run only the suggestion phase (reusing existing upstream results):
```bash
python "/project_class/run_pipeline.py" \
  --segments-json "your/knowledge_segments_global.json" \
  --writing-behavior-json "your/writing_behavior.json" \
  --output-dir "your_output_directory" \
  --only-suggestion
````
## 4. Evaluate Suggestion Quality (Optional)
Evaluation script:

/project_class/project_eval/eval.py

This script currently uses fixed paths defined under Path Config at the top of the file.

If you change the output directory, please modify the paths at the top of eval.py to match your directory before running:

```bash
python "/project_class/project_eval/eval.py"
````
Common outputs:

demo_icap_eval.json

demo_icap_eval.md

## 5. Script Composition by Folder
The following section explains the main working directories in your current project.
```bash
MAVE/
├── 📂 project_class (Core Engineering Pipeline)
│   ├── 📄 run_pipeline.py  <────── [Main Entry: Orchestrates all core modules]
│   │
│   ├── 📂 project_engage/ ───────► [Video Processing & Behavior Extraction] (Video splitting, frame extraction, face tracking, Action A)
│   ├── 📂 project-teacher/ ──────► [ICAP Pedagogical Analysis] (Knowledge points, CI inference, suggestion generation)
│   ├── 📂 project_eval/ ─────────► [Quality Evaluation] (Accuracy, Alignment, Actionability)
│   └── 📂 vis_result/ ───────────► [Visualization] (Trend charts, comparison charts, distribution plots)
│
├── 📂 prompt (Baseline Method)
│   └── 💡 Prompt-only ───────────► [Baseline suggestions and evaluation without optimization loops]
│
├── 📂 reflexion_my (Reflexion Experiments)
│   └── 🔄 run_reflexion_repo.py ─► [Generates transcript.jsonl with reflection signals]
│
├── 📂 textgrad_my (TextGrad Experiments)
│   └── 📉 trainwith_textgrad.py ─► [Transcript optimization based on text gradients]
│
└── 📂 ablation (Ablation Studies)
    ├── 📂 wo_observer/ ──────────► [Suggestions and evaluation without the Observer role]
    ├── 📂 wo_accessor/ ──────────► [Suggestions and evaluation without the Accessor role]
    └── 📂 wo_curator/ ───────────► [Suggestions and evaluation without the Curator role]
````

## 6. Your Shortest Path Now (Recommended)
Prepare the video + API key.

Run Whisper (or YouTube transcription) to produce knowledge_segments_global.json and writing_behavior.json.

Run run_pipeline.py to obtain icap_suggestions.json.

Run project_eval/eval.py when comparisons are needed.

## 7. Result Files Quick Reference
Suggestion Results (sug): icap_suggestions.json

Pipeline Run Summary: pipeline_run_summary.json

Evaluation Results: demo_icap_eval.json / demo_icap_eval.md
