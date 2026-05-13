# CuriousLLMs

Curiosity-driven RL on math problems with LoRA fine-tuning.

The `slurm` branch runs entirely on the Babel HPC cluster — no Tinker / cloud
dependency — via vLLM (inference) + PyTorch + PEFT (training). Two-stage
pipeline: optional format SFT cold-start, then RL with RND curiosity + PPO.

- **Operator guide:** [docs/pipeline.md](docs/pipeline.md) — what's ready, how
  to run each stage, chained recipe, CLI reference, troubleshooting.
- **SLURM-specific notes:** [slurm/README.md](slurm/README.md) — env patches
  (DeepGEMM, lora_shrink_op), memory rules, smoke tests.