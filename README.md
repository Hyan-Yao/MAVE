# MAVE: A Multi-Agent Framework for Grounded Pedagogical Feedback from Online Class Videos
This repository contains the implementation of **MAVE**, a collaborative multi-agent framework designed to transform raw multimodal classroom data into structured, 
pedagogically actionable optimization signals.
## Overview
Timely pedagogical feedback is critical for effective online instruction, yet existing AI-based educational assessment systems still struggle to generate actionable and trustworthy recommendations from authentic classroom environments. Major challenges include long-context noisy videos and weak grounding in real student behaviors. To address these challenges, we introduce **MAVE** (**M**ultimodal **A**gents for **V**ideo-based **E**ducational Feedback), a collaborative multi-agent framework for generating pedagogically grounded feedback from online classroom videos. Built upon the ICAP framework, MAVE formulates feedback generation as a discrepancy reasoning problem between instructors’ intended engagement objectives and students’ observed cognitive states. MAVE decomposes classroom understanding into specialized agents for instructional structuring, multimodal evidence extraction, engagement assessment, and pedagogical recommendation generation. By grounding feedback in temporally aligned multimodal classroom evidence, MAVE produces interpretable and actionable instructional recommendations. Experiments on real-world online classroom videos demonstrate that MAVE generates more pedagogically grounded and actionable feedback than existing LLM-based feedback generation baselines. 

---
## Framework Architecture
<img width="1422" height="623" alt="截屏2026-05-31 18 15 56" src="https://github.com/user-attachments/assets/ee29f3f5-6e83-48ee-862e-d74e00a29948" />

> *Figure: Overview of the proposed framework, including Instructional Context Recognition, Learner Engagement Quantification, and Feedback Generation.


---
*This project is part of ongoing research into fine-grained feedback-to-optimization paradigms in intelligent education.*
