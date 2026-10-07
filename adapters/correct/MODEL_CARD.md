# Lab 21 — Vietnamese customer-support triage LoRA

Author: Đào Ngọc Bình Thiên — 2A202602814. Run date: 2026-10-07.

This supplemental card describes the measured lab adapter. The PEFT-generated README is preserved unchanged for provenance.

- Base: `unsloth/Qwen3.5-4B`; source commit `d27c1c0`.
- Objective: supervised assistant-only loss on Vietnamese ticket → JSON triage (`intent`, `urgency`, `product`, `sentiment`).
- Dataset: lab seed corpus, 250 examples, 225 train / 25 validation, seed 42. Full evaluation uses 50 target tickets and 15 regression prompts.
- LoRA: text-decoder linear projections, rank 16, alpha 32, dropout 0; 32,464,896 trainable parameters.
- Training: 2 epochs / 30 planned optimizer steps; LR 1e-4, cosine scheduler, 3 warmup steps; microbatch 1, accumulation 16; max length 1024; fp16 on Tesla T4.
- Measured scores: target 0.970, regression 0.6556, format 1.000, generation latency 1371.2 ms/item in batch 4.
- Optimized-prompt base: target 0.765, regression 0.7911, format 1.000, latency 1024.6 ms/item.
- Regression gate: **FAILED**, because regression dropped 0.13556, beyond tolerance 0.020. This is an experimental adapter, not a validated general-purpose deployment.
- Limitations: synthetic small corpus, one seed, 15-item keyword-recall regression test, errors on low-urgency phrasing; no production evaluation or reasoning-collapse experiment.

Weights require the separate base model. See the submission report, original metrics and verification output for detailed evidence. No base weights, credential, or claim of a new model license is included.
