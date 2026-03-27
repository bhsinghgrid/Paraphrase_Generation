# Sanskrit Diffusion Model Optimization - Final Project Walkthrough

This document summarizes the engineering tasks, analytical findings, and final documentation deliverables for the Sanskrit Diffusion Paraphrasing project. 

## Project Objective
The goal was to optimize a D3PM-based (Discrete Diffusion) sequence model for Sanskrit text generation and provide comprehensive technical reports for five core tasks using real codebase implementations and empirical benchmarks.

## 🏁 Key Deliverables

All deliverables are provided in professional, academic-style PDF formats within their respective `TaskX` folders.

### 1. Task 1: KV Caching & Efficiency
- **Objective:** Slashed redundant encoding 
- **Achievement:** Implemented `generate_cached()` and a 50% semantic residual bottleneck.
- **Result:** **1.33x Speedup** and **24.6% RAM Reduction** on the T4 model.
- [Task1_Report.pdf](file:///Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/Task1/Task1_Report.pdf)

### 2. Task 2: Attention & Semantic Drift
- **Objective:** Track source-target alignment stability.
- **Achievement:** Registered forward hooks for cross-attention and computed BERTScore trajectories.
- **Result:** T4 configuration shows **|r| = 0.9472** correlation between TF-IDF importance and attention stability.
- [Task2_Report.pdf](file:///Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/Task2/Task2_Report.pdf)

### 3. Task 3: Concept Vector Extraction (PCA)
- **Objective:** Extract interpretable latent directions for diversity.
- **Achievement:** Performed PCA on 1000 hidden states; built steering pipeline.
- **Result:** **72.0% Variance Explained** but confirmed weak steering capability due to latent entanglement.
- [Task3_Report.pdf](file:///Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/Task3/Task3_Report.pdf)

### 4. Task 4: Multi-Step Ablation (Optimal Model Search)
- **Objective:** Identify the optimal diffusion step count (T).
- **Achievement:** Evaluated T={4, 8, 16, 32, 64}.
- **Finding:** **T=4 is the optimal model.** Higher step counts lead to noise accumulation and quality degradation in discrete diffusion.
- [Task4_Report.pdf](file:///Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/Task4/Task4_Report.pdf)

### 5. Task 5: Classifier-Free Guidance (CFG)
- **Objective:** Steer generation towards high-quality outputs.
- **Achievement:** Trained a secondary quality classifier (139K params).
- **Finding:** **Guidance failed.** Due to low training data (n=40), λ > 0 degraded performance. **Optimal setting: λ = 0.0.**
- [Task5_Report.pdf](file:///Users/bhsingh/Documents/Final_Paraphrase/Exclude_Negative/final_folder/Task5/Task5_Report.pdf)

---

## 🚀 Final Conclusion
The project establishes **T=4** as the high-performance baseline for Sanskrit diffusion. It provides a robust, cached inference pipeline that balances speed (1.33x gains) with semantic fidelity (0.94 attention correlation). 

**Optimization Status:** Complete.
