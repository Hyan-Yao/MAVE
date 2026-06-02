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

## Installation
Clone the repository and create the environment:

```bash
git clone https://github.com/Hyan-Yao/MAVE.git
cd mave
conda env create -f environment.yml
conda activate mave
````

---
*This project is part of ongoing research into fine-grained feedback-to-optimization paradigms in intelligent education.*
