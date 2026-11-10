"""The lean local RL loop: vLLM rollouts -> reward shaping -> KL -> Trainer step -> LoRA hot-swap.

SPMD layout (``WORLD_SIZE > 1``): every rank builds a ``Trainer`` and calls ``trainer.logprobs`` /
``trainer.train_step`` / ``save_checkpoint`` in the same order. Only rank 0 samples from vLLM,
shapes rewards (Gemini judge, SemanticRND on rank 0's GPU), syncs the adapter, evaluates, logs
and writes checkpoints; it broadcasts the assembled ``list[tinker.Datum]`` to the other ranks.

Preemption (Babel gives <= 120 s from SIGTERM to SIGKILL): a ``PreemptionFlag`` is installed on
every rank. At each safe point (after the rollouts, before the KL/train phase, right after
``train_step``) the ranks agree on the flag (all-reduce MAX). An in-progress step is abandoned
(its rollouts discarded, RND state rolled back to the start of the step), the last completed
step is checkpointed if it is not already, and ``run`` returns normally (exit 0).
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import os
import shutil
import signal
import time
from dataclasses import dataclass
from typing import Any, Sequence

import tinker
import torch
from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.metrics import compute_post_kl, incorporate_kl_penalty
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed

from lean_rl.checkpoint import (
    LoopState,
    PreemptionFlag,
    load_latest,
    restore_checkpoint,
    save_checkpoint,
)
from lean_rl.config import LeanRLConfig
from lean_rl.shaping import get_memory_usage, shape_rewards
from lean_rl.trainer import Trainer

logger = logging.getLogger(__name__)

# Tinker optimizer parity (curiosity_train.optim_step): Adam b1 0.9, b2 0.95, eps 1e-8, no WD.
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.95, 1e-8
_POLL_S = 0.5  # how often rollouts check the preemption flag


class PrecomputedLogprobs:
    """Duck-types ``compute_logprobs_async`` for cookbook KL code from trainer logprobs.

    Cookbook ``incorporate_kl_penalty`` / ``compute_post_kl`` ask for the logprobs of
    ``model_input + [last target]``; the trainer computed them for every datum in one batched,
    SPMD-safe call, so this just looks them up (``[None] + target logprobs``).
    """

    def __init__(self, data: Sequence[tinker.Datum], logprobs: Sequence[torch.Tensor]):
        self._by_tokens: dict[tuple[int, ...], list[float | None]] = {}
        for datum, lp in zip(data, logprobs, strict=True):
            last = int(datum.loss_fn_inputs["target_tokens"].to_torch()[-1])
            full = tuple(datum.model_input.to_ints()) + (last,)
            self._by_tokens[full] = [None] + [float(x) for x in lp.tolist()]

    async def compute_logprobs_async(self, model_input: tinker.ModelInput) -> list[float | None]:
        return self._by_tokens[tuple(model_input.to_ints())]


@dataclass
class _Rank:
    rank: int = 0
    world_size: int = 1
    is_main: bool = True
    info: Any = None  # lean_rl.dist.DistInfo when multi-rank


class _Comm:
    """Collectives used by the loop; no-ops in a single process."""

    def __init__(self, r: _Rank, device: str):
        self.r = r
        self.device = device
        self._ldist = None
        if r.info is not None:
            from lean_rl import dist as ldist

            self._ldist = ldist

    def broadcast(self, obj: Any) -> Any:
        return obj if self._ldist is None else self._ldist.broadcast_object(obj, src=0)

    def agree(self, local: bool) -> bool:
        """True on every rank if any rank says True (all-reduce MAX)."""
        if self._ldist is None:
            return bool(local)
        import torch.distributed as tdist

        t = torch.tensor([1 if local else 0], dtype=torch.int32, device=self.device)
        tdist.all_reduce(t, op=tdist.ReduceOp.MAX)
        return bool(t.item())


def _init_rank() -> tuple[_Rank, str]:
    world = int(os.environ.get("WORLD_SIZE", "1") or "1")
    cuda = torch.cuda.is_available()
    if world <= 1:
        return _Rank(), "cuda" if cuda else "cpu"
    from lean_rl import dist as ldist

    info = ldist.init_dist()
    if info is None:
        return _Rank(), "cuda" if cuda else "cpu"
    device = f"cuda:{info.local_rank}" if cuda else "cpu"
    if cuda:
        torch.cuda.set_device(info.local_rank)
    return _Rank(info.rank, info.world_size, info.is_main, info), device


def _validate(cfg: LeanRLConfig) -> None:
    if cfg.kl_reference == "init" and not cfg.init_from_adapter:
        raise ValueError("kl_reference=init requires init_from_adapter to be set.")
    if cfg.save_every <= 0:
        raise ValueError("save_every must be >= 1")


# ---------------------------------------------------------------------------- RND (rank 0)
def _make_rnd(cfg: LeanRLConfig, device: str, batch_size: int):
    if not cfg.use_rnd_curiosity:
        logger.info("RND curiosity rewards disabled (use_rnd_curiosity=False)")
        return None, None
    from rnd import RNDBuffer, SemanticRND

    # Same setup as curiosity_train.main; concat_before_emb=False is required by the buffer.
    model = SemanticRND(
        encoder_model_name="Alibaba-NLP/gte-modernbert-base",
        target_layers=cfg.target_layers,
        predictor_layers=cfg.predictor_layers,
        device=device,
        max_length=8192,
        concat_before_emb=False,
        separator=" [SEP] ",
        normalize_embeddings=True,
        embedding_load_file="embeddings/embeddings_gte_10000.pkl",
    )
    model._optimizer = torch.optim.Adam(model.rnd.predictor_parameters(), lr=cfg.rnd_learning_rate)
    assert model.embedding_dim is not None
    buffer = RNDBuffer(
        max_batches=cfg.rnd_buffer_size_multiplier,
        batch_size=batch_size,
        group_size=cfg.group_size,
        embedding_dim=model.embedding_dim,
        device=device,
    )
    logger.info(f"Initialized SemanticRND on {device} (lr={cfg.rnd_learning_rate}) + buffer")
    return model, buffer


def _rnd_state(model, buffer) -> dict | None:
    if model is None or buffer is None:
        return None
    mb, bb = io.BytesIO(), io.BytesIO()
    model.save_state(mb)
    buffer.save_state(bb)
    return {"model": mb.getvalue(), "buffer": bb.getvalue()}


def _load_rnd_state(model, buffer, state: dict | None) -> None:
    if model is None or buffer is None:
        return
    if state is None:
        logger.warning("Checkpoint has no RND state; RND starts fresh")
        return
    model.load_state(io.BytesIO(state["model"]))
    buffer.load_state(io.BytesIO(state["buffer"]))


def _make_judge(cfg: LeanRLConfig):
    if not cfg.use_reasoning_rewards:
        return None
    from llm_as_a_judge import GEMINI_MATH_JUDGE_SYSTEM_PROMPT, GeminiJudge

    logger.info(f"Initialized GeminiJudge (coef={cfg.reasoning_reward_coef})")
    return GeminiJudge(model_name="gemini-2.5-flash-lite", system_prompt=GEMINI_MATH_JUDGE_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------- sampling
async def _rollouts(
    sampling_client, builders: Sequence[EnvGroupBuilder], max_tokens: int, flag: PreemptionFlag
) -> list[TrajectoryGroup] | None:
    """All group rollouts concurrently; None (tasks cancelled) if preemption is requested."""
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)
    tasks = [asyncio.create_task(do_group_rollout(b, policy)) for b in builders]
    pending = set(tasks)
    while pending:
        if flag.requested:
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            return None
        _, pending = await asyncio.wait(pending, timeout=_POLL_S)
    if flag.requested:
        return None
    return [t.result() for t in tasks]


def _make_evaluators(test_datasets, max_tokens: int) -> list[RLTestSetEvaluator]:
    if test_datasets is None:
        return []
    if isinstance(test_datasets, list):
        names = ["math_test", "deepmath_test"]
        return [
            RLTestSetEvaluator(ds, max_tokens=max_tokens, name=names[i] if i < len(names) else f"test_{i}")
            for i, ds in enumerate(test_datasets)
        ]
    return [RLTestSetEvaluator(test_datasets, max_tokens=max_tokens)]


async def _run_evals(evaluators, sampling_client) -> dict[str, Any]:
    results = await asyncio.gather(*[ev(sampling_client) for ev in evaluators])
    out: dict[str, Any] = {}
    for r in results:
        out.update({f"test/{k}": v for k, v in r.items()})
    return out


# ---------------------------------------------------------------------------- the loop
class _Loop:
    def __init__(self, cfg: LeanRLConfig, transport=None):
        self.cfg = cfg
        self.transport = transport  # test hook: httpx transport for the vLLM client
        self.flag = PreemptionFlag()
        self.ml_logger = None
        self.last_saved_step = -1

    # ------------------------------------------------------------------ setup
    def _setup_logging(self) -> None:
        cfg = self.cfg
        if not self.r.is_main:
            logging.basicConfig(level=logging.INFO, format=f"[rank {self.r.rank}] %(name)s %(levelname)s %(message)s")
            return
        found = load_latest(cfg.log_path)
        run_id = found[1].wandb_run_id if found else None
        if run_id and cfg.wandb_project:
            os.environ["WANDB_RUN_ID"] = run_id
            os.environ.setdefault("WANDB_RESUME", "allow")
        self.ml_logger = ml_log.setup_logging(
            log_dir=cfg.log_path,
            wandb_project=cfg.wandb_project,
            wandb_name=cfg.wandb_name,
            config=cfg,
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    def _wandb_run_id(self) -> str | None:
        for lg in getattr(self.ml_logger, "loggers", []) or []:
            run = getattr(lg, "run", None)
            if run is not None and getattr(run, "id", None):
                return str(run.id)
        return None

    def _build_trainer(self) -> Trainer:
        cfg = self.cfg
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        # Trainer.__init__ logs the Tinker-parity warnings (log_parity_warnings) once.
        return Trainer(
            cfg.model_name,
            rank=cfg.lora_rank,
            device=self.device,
            dtype=dtype,
            gpt_oss_lm_head=cfg.gpt_oss_lm_head,
            dist=self.r.info,
            fsdp=cfg.fsdp,
            micro_batch_tokens=cfg.micro_batch_tokens,
            attn_implementation=cfg.attn_implementation,
        )

    async def _setup_data(self):
        from math_train import get_dataset_builder

        cfg = self.cfg
        renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)
        builder = get_dataset_builder(
            env=cfg.env,
            batch_size=cfg.groups_per_batch,
            model_name=cfg.model_name,
            renderer_name=renderer_name,
            group_size=cfg.group_size,
            seed=cfg.seed,
            math_train_size=cfg.math_train_size,
            deepmath_train_size=cfg.deepmath_train_size,
            deepmath_test_size=cfg.deepmath_test_size,
            deepmath_seed=cfg.deepmath_seed,
        )
        return await builder()

    # ------------------------------------------------------------------ checkpoint / adapter
    def _save(self, step: int, rnd_state: dict | None) -> None:
        """All ranks call this (rank 0 writes)."""
        if step <= self.last_saved_step:
            return
        path = save_checkpoint(
            self.cfg.log_path,
            step,
            self.trainer,
            LoopState(step=step, data_position=step, wandb_run_id=self._wandb_run_id()),
            rnd_state if self.r.is_main else None,
            keep=self.cfg.keep_checkpoints,
            is_main=self.r.is_main,
        )
        self.last_saved_step = step
        if self.r.is_main:
            logger.info(f"Saved checkpoint for step {step} at {path}")

    def _sync_adapter(self, step: int) -> None:
        """Rank 0: write the current adapter and hot-swap it into every vLLM server."""
        from local_backend.weight_sync import sync_lora_to_vllm

        root = os.path.join(self.cfg.log_path, "serving_adapter")
        path = os.path.join(root, f"step_{step:06d}")
        name = f"lean_step_{step:06d}"
        self.trainer.save_adapter(path)
        sync_lora_to_vllm(path, name, self.cfg.vllm_urls, prev_name=self.sampling_client.adapter_name)
        self.sampling_client.adapter_name = name
        for d in os.listdir(root):  # vLLM has loaded the new one; older dirs are unused
            if d != os.path.basename(path):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

    # ------------------------------------------------------------------ main
    async def run(self) -> None:
        cfg = self.cfg
        _validate(cfg)
        self.r, self.device = _init_rank()
        self.comm = _Comm(self.r, self.device)
        old_handlers = {s: signal.getsignal(s) for s in PreemptionFlag.SIGNALS}
        self.flag.install()
        try:
            self._setup_logging()
            # The Trainer is built before anything else so its parity warnings head the log.
            self.trainer = self._build_trainer()
            await self._run()
        finally:
            for s, h in old_handlers.items():
                signal.signal(s, h)
            if self.ml_logger is not None:
                self.ml_logger.close()
        if self.r.info is not None:  # clean exit only; on error the launcher tears ranks down
            import torch.distributed as tdist

            if tdist.is_initialized():
                tdist.destroy_process_group()

    async def _run(self) -> None:
        cfg = self.cfg
        is_main = self.r.is_main
        dataset, test_datasets = await self._setup_data()
        n_total = len(dataset) if cfg.num_batches is None else min(len(dataset), cfg.num_batches)

        novelty_model = rnd_buffer = gemini_judge = None
        tokenizer = None
        evaluators: list = []
        if is_main:
            if not cfg.vllm_urls:
                raise ValueError("vllm_urls is required (comma-separated vLLM base URLs)")
            from local_backend.sampling_client import LocalSamplingClient

            tokenizer = get_tokenizer(cfg.model_name)
            novelty_model, rnd_buffer = _make_rnd(cfg, self.device, len(dataset.get_batch(0)))
            gemini_judge = _make_judge(cfg)
            evaluators = _make_evaluators(test_datasets, cfg.max_tokens)
            # adapter_name None -> the base model; set by _sync_adapter after each step.
            self.sampling_client = LocalSamplingClient(
                vllm_urls=cfg.vllm_urls,
                base_model_name=cfg.model_name,
                tokenizer=tokenizer,
                transport=self.transport,
            )

        # Resume > init_from_adapter > fresh.
        start = 0
        found = load_latest(cfg.log_path)
        if found is not None:
            path, state = found
            rnd_state = restore_checkpoint(path, self.trainer)
            if is_main:
                _load_rnd_state(novelty_model, rnd_buffer, rnd_state)
            start = state.data_position
            self.last_saved_step = state.step
            logger.info(f"Resumed from {path} (step {state.step})")
        elif cfg.init_from_adapter:
            self.trainer.load_adapter(cfg.init_from_adapter)
            self.trainer.snapshot_init_adapter()  # KL reference for kl_reference=init
            logger.info(f"Warm-started adapter from {cfg.init_from_adapter} (fresh optimizer)")
        # (On resume the init snapshot comes back from the checkpoint.)
        if is_main and (found is not None or cfg.init_from_adapter):
            self._sync_adapter(start)  # vLLM must sample from the restored / warm-start policy
        if start >= n_total:
            logger.info(f"Nothing to do: step {start} >= {n_total} batches")
            return
        logger.info(f"Training batches {start}..{n_total - 1}")

        adam = tinker.AdamParams(
            learning_rate=cfg.learning_rate, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=ADAM_EPS
        )
        for i_batch in range(start, n_total):
            if not await self._step(
                i_batch, n_total, dataset, adam, tokenizer, novelty_model, rnd_buffer, gemini_judge, evaluators
            ):
                return  # preempted; checkpoint written
        self._save(n_total, _rnd_state(novelty_model, rnd_buffer) if is_main else None)
        logger.info("Training completed")

    async def _step(
        self, i_batch, n_total, dataset, adam, tokenizer, novelty_model, rnd_buffer, gemini_judge, evaluators
    ) -> bool:
        """One RL step. Returns False if preempted (after saving the last completed step)."""
        cfg = self.cfg
        is_main = self.r.is_main
        t0 = time.time()
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / n_total,
        }
        data: list[tinker.Datum] | None = None
        rnd_before = None
        abandoned = False

        if is_main:
            if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0 and evaluators:
                with timed("run_evals", metrics):
                    metrics.update(await _run_evals(evaluators, self.sampling_client))
            builders = dataset.get_batch(i_batch)
            with timed("sample", metrics):
                groups = None if self.flag.requested else await _rollouts(
                    self.sampling_client, builders, cfg.max_tokens, self.flag
                )
            if groups is None:
                abandoned = True
            else:
                # Pre-shaping RND state: what to checkpoint if this step is abandoned below.
                rnd_before = _rnd_state(novelty_model, rnd_buffer)
                metrics.update(
                    await shape_rewards(
                        builders,
                        groups,
                        tokenizer,
                        i_batch=i_batch,
                        gemini_judge=gemini_judge,
                        reasoning_reward_coef=cfg.reasoning_reward_coef,
                        novelty_model=novelty_model,
                        rnd_buffer=rnd_buffer,
                        intrinsic_reward_coef=cfg.curiosity_reward_coef,
                        rnd_update_steps=cfg.rnd_update_steps,
                        rnd_minibatch_size=cfg.rnd_minibatch_size,
                        penalize_incorrect_novelty=cfg.penalize_incorrect_novelty,
                        correctness_threshold=cfg.correctness_threshold,
                        curiosity_warmup_batches=cfg.curiosity_warmup_batches,
                        entropy_bonus_coef=cfg.entropy_bonus_coef,
                    )
                )
                with timed("assemble_training_data", metrics):
                    data, _ = assemble_training_data(groups, compute_advantages(groups))

        # Safe point 1: abandon the step before any trainer work.
        if self.comm.agree(abandoned or self.flag.requested):
            logger.info(f"Preemption requested during step {i_batch}: abandoning it")
            if is_main and rnd_before is None:
                rnd_before = _rnd_state(novelty_model, rnd_buffer)
            self._save(i_batch, rnd_before)
            return False

        data = self.comm.broadcast(data)
        if cfg.kl_penalty_coef > 0:
            with timed("kl_vs_ref", metrics):
                ref_lp = self.trainer.logprobs(data, cfg.kl_reference)  # all ranks (SPMD)
                if is_main:
                    metrics.update(
                        await incorporate_kl_penalty(
                            data, PrecomputedLogprobs(data, ref_lp), cfg.kl_penalty_coef, cfg.kl_discount_factor
                        )
                    )
            data = self.comm.broadcast(data)  # KL-adjusted advantages

        with timed("train", metrics):
            metrics.update(self.trainer.train_step(data, cfg.loss_fn, adam))  # all ranks
        step = i_batch + 1

        if cfg.compute_post_kl:
            with timed("compute_post_kl", metrics):
                post_lp = self.trainer.logprobs(data, "policy")  # all ranks
                if is_main:
                    metrics.update(await compute_post_kl(data, PrecomputedLogprobs(data, post_lp)))

        # Safe point 2: the step is complete; save it and stop if preempted.
        preempt = self.comm.agree(self.flag.requested)
        if is_main and not preempt:
            with timed("sync_adapter", metrics):
                self._sync_adapter(step)
        if is_main:
            metrics["time/total"] = time.time() - t0
            mem = get_memory_usage()
            metrics["memory/cpu_rss_mb"] = mem["cpu_rss_mb"]
            if "gpu_allocated_mb" in mem:
                metrics["memory/gpu_allocated_mb"] = mem["gpu_allocated_mb"]
            self.ml_logger.log_metrics(metrics, step=i_batch)
        if preempt or step % cfg.save_every == 0:
            self._save(step, _rnd_state(novelty_model, rnd_buffer) if is_main else None)
        if preempt:
            logger.info(f"Preemption requested: saved step {step}, exiting")
            return False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True


async def run(cfg: LeanRLConfig, *, transport=None) -> None:
    """Run (or resume) training. ``transport`` is a test hook for the vLLM HTTP client."""
    await _Loop(cfg, transport=transport).run()
