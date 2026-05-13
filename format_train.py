"""
Format SFT cold-start training.

DeepSeek-R1-style pre-RL supervised stage: response-masked NLL on a HuggingFace
dataset of format-correct (problem, solution) pairs. Produces a LoRA adapter
that you then warm-start RL from via `math_train.py init_from_adapter=...`.

Run via `slurm/sbatch_format.sh` on a single GPU.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import chz
import tinker

from tinker_cookbook import cli_utils

from local_backend.format_dataset import build_format_data, iter_batches
from local_backend.service_client import LocalServiceClient

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """CLI configuration for the format SFT cold-start stage."""

    # Model
    model_name: str = "meta-llama/Llama-3.2-3B"
    lora_rank: int = 32

    # Dataset
    dataset_name: str = "AI-MO/NuminaMath-CoT"
    dataset_split: str = "train"
    dataset_config: str | None = None
    problem_column: str = "problem"
    solution_column: str = "solution"
    max_rows: int | None = None  # cap dataset size; None = use all rows
    max_seq_len: int = 4096

    # Optimizer
    batch_size: int = 8
    num_steps: int = 500
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    seed: int = 0

    # Checkpointing / logging
    log_path: str | None = None
    save_every: int = 100
    log_every: int = 1
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Infrastructure
    base_url: str | None = None  # vLLM URL; only needed for the dummy service connection

    # wandb
    wandb_project: str | None = "rnd_train"
    wandb_entity: str | None = "CuriousLMs"
    wandb_name: str | None = None


async def main(cfg: CLIConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if cfg.num_steps <= 0:
        raise ValueError(f"num_steps must be > 0; got {cfg.num_steps}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0; got {cfg.batch_size}")

    # --- log path / wandb name ------------------------------------------------
    short_model = cfg.model_name.replace("/", "-")
    short_ds = cfg.dataset_name.replace("/", "-")
    run_tag = (
        f"format-{short_model}-{cfg.lora_rank}rank-{short_ds}-{cfg.num_steps}steps"
        f"-{cfg.learning_rate}lr-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_path = cfg.log_path or f"/data/hf_cache/ksnair/CuriousLLMs_logs/{run_tag}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)
    logger.info("log_path=%s", log_path)

    # --- wandb (optional) -----------------------------------------------------
    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_name or run_tag,
                dir=log_path,
                config={k: getattr(cfg, k) for k in vars(cfg).keys() if not k.startswith("_")},
            )
        except Exception as e:
            logger.warning("wandb init failed: %s; continuing without wandb", e)

    # --- build the SFT dataset (host-side tokenization) -----------------------
    logger.info("Building format dataset...")
    datums = build_format_data(
        model_name=cfg.model_name,
        dataset_name=cfg.dataset_name,
        dataset_split=cfg.dataset_split,
        dataset_config=cfg.dataset_config,
        problem_column=cfg.problem_column,
        solution_column=cfg.solution_column,
        max_seq_len=cfg.max_seq_len,
        max_rows=cfg.max_rows,
        seed=cfg.seed,
    )
    logger.info("Built %d Datums", len(datums))

    # --- training client ------------------------------------------------------
    service_client = LocalServiceClient(
        vllm_url=cfg.base_url or os.environ.get("VLLM_URL"),
        log_path=log_path,
        base_model_name=cfg.model_name,
    )
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    # --- training loop --------------------------------------------------------
    metrics_path = os.path.join(log_path, "format_metrics.jsonl")
    batches = iter_batches(datums, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed)
    # tinker.AdamParams is a frozen pydantic model — must pass everything in
    # the constructor. (setattr raises ValidationError on the frozen instance.)
    adam_params = tinker.AdamParams(
        learning_rate=cfg.learning_rate,
        beta1=cfg.adam_beta1,
        beta2=cfg.adam_beta2,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        grad_clip_norm=cfg.grad_clip_norm,
    )

    t_start = time.time()
    final_path: Optional[str] = None
    with open(metrics_path, "a") as metrics_f:
        for step in range(1, cfg.num_steps + 1):
            batch = next(batches)
            t0 = time.time()
            fb_fut = await training_client.forward_backward_async(batch, loss_fn="nll")
            fb = await fb_fut.result_async()
            opt_fut = await training_client.optim_step_async(adam_params)
            _ = await opt_fut.result_async()
            dt = time.time() - t0
            loss = float(fb.metrics.get("loss", float("nan")))
            n_resp = float(fb.metrics.get("response_tokens", 0.0))

            if step % cfg.log_every == 0 or step == 1:
                logger.info(
                    "step %d/%d  loss=%.4f  response_tokens=%.0f  step_time=%.2fs",
                    step, cfg.num_steps, loss, n_resp, dt,
                )
            metrics_f.write(json.dumps({
                "step": step, "loss": loss, "response_tokens": n_resp, "step_time": dt,
            }) + "\n")
            metrics_f.flush()
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss, "train/response_tokens": n_resp,
                               "train/step_time": dt}, step=step)

            if step % cfg.save_every == 0:
                save_fut = await training_client.save_state_async(f"step_{step:06d}")
                saved = await save_fut.result_async()
                logger.info("saved checkpoint @ step %d -> %s", step, saved.path)

    # Final save: stable name so the operator can pipe it into INIT_FROM_ADAPTER.
    save_fut = await training_client.save_state_async("format_final")
    saved = await save_fut.result_async()
    final_path = saved.path
    dt_total = time.time() - t_start
    logger.info("DONE: %d steps in %.1fs  -> format_final at %s", cfg.num_steps, dt_total, final_path)
    print(f"FORMAT_FINAL_PATH={final_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli_config))
