# Sanskrit Diffusion Model Architecture

This diagram visualizes the core transformer-based D3PM architecture and highlights how the five engineering tasks modify or analyze the generation pipeline.

```mermaid
graph TD
    subgraph "Phase 1: Source Encoding"
        A["Source (Sanskrit)"] --> B["Source Embedding"]
        B --> C["Transformer Encoder Blocks"]
        C --> D["Task 1: Semantic Residual Projection (50%)"]
        D --> E["Task 1: KV Cache (Keys/Values)"]
    end

    subgraph "Phase 2: Iterative Denoising (Diffusion)"
        F["Noisy Latent (x_T)"] --> G["Task 4: Step Scheduler (T=4 to T=64)"]
        G --> H["Transformer Decoder Block"]
        E -.-> |"Cross-Attention Injection"| H
        H --> I["Task 2: Attention Hook Capture"]
        I --> J["Task 3: Hidden State PCA Analysis"]
        
        subgraph "Guidance Loop (Task 5)"
            K["Task 5: Quality Classifier (139K)"]
            J -.-> |"Hidden State"| K
            K -.-> |"Logit Gradient (λ)"| L["Logit Steering Module"]
        end
        
        H --> L
        L --> M["Predicted x_0 Logits"]
        M --> N["Categorical Sampling"]
        N --> |"Next Step (x_t-1)"| G
    end

    subgraph "Phase 3: Output & Evaluation"
        N --> |"Final Step (t=0)"| O["Transliterated/Paraphrased Output"]
        O --> P["BERTScore & CER Metrics"]
        P --> Q["Task 4: Optimal Model Selection (T=4)"]
    end

    %% Task Links
    style D fill:#f9f,stroke:#333Msg
    style E fill:#f9f,stroke:#333
    style I fill:#bbf,stroke:#333
    style J fill:#bfb,stroke:#333
    style G fill:#ffb,stroke:#333
    style K fill:#fbb,stroke:#333
```

## Task Mapping Summary

| Task | Component | Description |
| :--- | :--- | :--- |
| **Task 1** | **KV Cache** | Pre-computes source representations to avoid redundant $O(T)$ encoding. |
| **Task 2** | **Attention Hooks** | Captures cross-attention weights to measure semantic drift and TF-IDF correlation. |
| **Task 3** | **PCA Steering** | extracts concept vectors from hidden states to control output diversity along $\alpha$ spectrum. |
| **Task 4** | **Ablation Studies** | Benchmarks $T$ across a scale to find the quality 'knee' (Optimal $T=4$). |
| **Task 5** | **CFG Guidance** | Injects classifier gradients into logits to steer generation toward high-BERTScore states. |

---
*Note: Darker labels represent task-specific modules added during the optimization phase.*
