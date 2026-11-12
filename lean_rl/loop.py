"""The lean local RL loop: vLLM rollouts -> reward shaping -> KL -> Trainer step -> LoRA hot-swap.

SPMD layout (``WORLD_SIZE > 1``): every rank builds a ``Trainer`` and makes every trainer call
(``logprobs``, ``train_step``, ``save_adapter``, ``snapshot_init_adapter``, ``load_adapter``,
``restore_checkpoint`` / ``save_checkpoint``) in the same order -- see the contract in
``lean_rl.trainer``. Only rank 0 samples from vLLM,
shapes rewards (Gemini judge, SemanticRND on rank 0's GPU), syncs the adapter, evaluates, logs
and writes checkpoints; it broadcasts the assembled ``list[tinker.Datum]`` to the other ranks.

Preemption (Babel gives <= 120 s from SIGTERM to SIGKILL, and SIGTERMs vLLM too): a
``PreemptionFlag`` is installed on every rank. Rank 0 polls it during evals and rollouts; the
ranks agree on it (``lean_rl.dist.any_true``: all-reduce MAX) after the rollouts, after the
KL reference logprobs, after every ``train_step`` micro-batch (``Trainer.train_step(...,
should_stop=)``), at the end of the step and after the vLLM adapter sync. A vLLM call that
fails while the flag is (or within ``_PREEMPT_GRACE_S`` becomes) set counts as preemption, not
an error. An in-progress step is abandoned (rollouts discarded, grads dropped without an
optimizer step, RND state rolled back to the start of the step), the last completed step is
checkpointed if it is not already, and ``run`` returns normally (exit 0) on every rank.

Resume: rank 0 clears stale ``step_*.tmp`` dirs and picks the newest checkpoint that loads
(``load_latest``); the choice is broadcast so every rank restores the same path.
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
    remove_stale_tmp,
    restore_checkpoint,
    save_checkpoint,
)
from lean_rl.config import LeanRLConfig
from lean_rl.shaping import get_memory_usage, shape_rewards
from lean_rl.trainer import Trainer

logger = logging.getLogger(__name__)

# Tinker optimizer parity (curiosity_train.optim_step): Adam b1 0.9, b2 0.95, eps 1e-8, no WD.
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.95, 1e-8
_POLL_S = 0.5  # how often rollouts / evals check the preemption flag
# After a vLLM call fails, how long to wait for our own SIGTERM: Slurm signals vLLM and the
# trainer at about the same time, so vLLM can die a moment before our flag is set.
_PREEMPT_GRACE_S = 10.0


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
        """True on every rank if any rank says True (all-reduce MAX on the gloo group)."""
        if self._ldist is None:
            return bool(local)
        return self._ldist.any_true(local)


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
def _preempted_after_error(flag: PreemptionFlag, exc: BaseException) -> bool:
    """A vLLM call raised ``exc``: True if preemption is (or within the grace period becomes)
    requested, i.e. vLLM was most likely killed by the same Slurm preemption."""
    deadline = time.monotonic() + _PREEMPT_GRACE_S
    while not flag.requested and time.monotonic() < deadline:
        time.sleep(0.05)  # the signal handler runs in this (main) thread meanwhile
    if flag.requested:
        logger.warning(f"vLLM call failed during preemption ({exc!r}); treating it as preemption")
    return flag.requested


async def _gather_unless_preempted(coros: Sequence[Any], flag: PreemptionFlag) -> list | None:
    """Run ``coros`` concurrently, polling the flag. None (all tasks cancelled) if preemption
    is requested meanwhile or a task fails because of it; any other failure is re-raised."""
    tasks = [asyncio.create_task(c) for c in coros]
    pending = set(tasks)
    try:
        while pending:
            if flag.requested:
                return None
            done, pending = await asyncio.wait(pending, timeout=_POLL_S)
            for t in done:
                exc = None if t.cancelled() else t.exception()
                if exc is not None:
                    if _preempted_after_error(flag, exc):
                        return None
                    raise exc
        return None if flag.requested else [t.result() for t in tasks]
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _rollouts(
    sampling_client, builders: Sequence[EnvGroupBuilder], max_tokens: int, flag: PreemptionFlag
) -> list[TrajectoryGroup] | None:
    """All group rollouts concurrently; None (tasks cancelled) if preempted."""
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)
    return await _gather_unless_preempted([do_group_rollout(b, policy) for b in builders], flag)


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


async def _run_evals(evaluators, sampling_client, flag: PreemptionFlag) -> dict[str, Any] | None:
    """All evaluators concurrently; None (abandoned) if preempted."""
    results = await _gather_unless_preempted([ev(sampling_client) for ev in evaluators], flag)
    if results is None:
        return None
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
    def _find_resume(self) -> tuple[str, LoopState] | None:
        """Rank 0 clears stale ``.tmp`` checkpoint dirs and picks the checkpoint to resume from
        (newest that loads); every rank gets rank 0's choice (or its error)."""
        found: Any = None
        if self.r.is_main:
            try:
                remove_stale_tmp(self.cfg.log_path)
                found = load_latest(self.cfg.log_path)
            except Exception as e:  # noqa: BLE001 -- re-raised on every rank below
                found = e
        found = self.comm.broadcast(found)
        if isinstance(found, Exception):
            raise found
        return found

    def _setup_logging(self) -> None:
        cfg = self.cfg
        if not self.r.is_main:
            logging.basicConfig(level=logging.INFO, format=f"[rank {self.r.rank}] %(name)s %(levelname)s %(message)s")
            return
        found = self._resume
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

    def _sync_adapter(self, step: int) -> bool:
        """All ranks: write the current adapter (rank 0 writes; FSDP gathers on every rank);
        rank 0 then hot-swaps it into every vLLM server. Returns False on every rank if the
        sync failed because of preemption (vLLM is SIGTERMed with us)."""
        from local_backend.weight_sync import sync_lora_to_vllm

        root = os.path.join(self.cfg.log_path, "serving_adapter")
        path = os.path.join(root, f"step_{step:06d}")
        name = f"lean_step_{step:06d}"
        self.trainer.save_adapter(path)  # collective under FSDP
        preempted = False
        if self.r.is_main:
            try:
                sync_lora_to_vllm(
                    path, name, self.cfg.vllm_urls, prev_name=self.sampling_client.adapter_name
                )
            except Exception as e:  # noqa: BLE001
                if not _preempted_after_error(self.flag, e):
                    raise
                preempted = True
            else:
                self.sampling_client.adapter_name = name
                for d in os.listdir(root):  # vLLM has loaded the new one; older dirs are unused
                    if d != os.path.basename(path):
                        shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        return not self.comm.agree(preempted)

    # ------------------------------------------------------------------ main
    async def run(self) -> None:
        cfg = self.cfg
        _validate(cfg)
        self.r, self.device = _init_rank()
        multi = self.r.info is not None
        if multi:
            from lean_rl import dist as ldist

            # Fail fast: an unhandled error on any rank hard-exits every rank within seconds
            # instead of leaving the others blocked in a collective until the dist timeout.
            ldist.start_watchdog()
        old_handlers = {s: signal.getsignal(s) for s in PreemptionFlag.SIGNALS}
        try:
            self.comm = _Comm(self.r, self.device)
            self.flag.install()
            self._resume = self._find_resume()
            self._setup_logging()
            # The Trainer is built before anything else so its parity warnings head the log.
            self.trainer = self._build_trainer()
            await self._run()
        except BaseException as e:
            if multi:
                ldist.abort(e)  # never returns; before the cleanup below, which could block
            raise
        finally:
            for s, h in old_handlers.items():
                signal.signal(s, h)
            if self.ml_logger is not None:
                self.ml_logger.close()
        if multi:  # clean exit (completed or preempted): stop watchdogs, barrier, destroy
            ldist.shutdown()

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
        found = self._resume  # rank 0's choice, the same path on every rank
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
        # vLLM must sample from the restored / warm-start policy. All ranks (save_adapter is
        # collective under FSDP).
        if (found is not None or cfg.init_from_adapter) and not self._sync_adapter(start):
            logger.info("Preempted during the startup adapter sync; exiting")
            return  # nothing new to save
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
            if (
                cfg.eval_every > 0 and i_batch % cfg.eval_every == 0 and evaluators
                and not self.flag.requested
            ):
                with timed("run_evals", metrics):
                    evals = await _run_evals(evaluators, self.sampling_client, self.flag)
                if evals is not None:
                    metrics.update(evals)
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
        if is_main and rnd_before is None:
            rnd_before = _rnd_state(novelty_model, rnd_buffer)
        if self.comm.agree(abandoned or self.flag.requested):
            return self._abandon(i_batch, rnd_before)

        data = self.comm.broadcast(data)
        if cfg.kl_penalty_coef > 0:
            with timed("kl_vs_ref", metrics):
                ref_lp = self.trainer.logprobs(data, cfg.kl_reference)  # all ranks (SPMD)
                # Safe point 2: the reference logprobs can take minutes on large models.
                if self.comm.agree(self.flag.requested):
                    return self._abandon(i_batch, rnd_before)
                if is_main:
                    metrics.update(
                        await incorporate_kl_penalty(
                            data, PrecomputedLogprobs(data, ref_lp), cfg.kl_penalty_coef, cfg.kl_discount_factor
                        )
                    )
            data = self.comm.broadcast(data)  # KL-adjusted advantages

        with timed("train", metrics):
            # All ranks; the ranks agree on the flag after every micro-batch (safe point 3).
            train_metrics = self.trainer.train_step(
                data, cfg.loss_fn, adam, should_stop=lambda: self.flag.requested
            )
        if train_metrics is None:  # abandoned on every rank: no optimizer step was taken
            return self._abandon(i_batch, rnd_before)
        metrics.update(train_metrics)
        step = i_batch + 1

        if cfg.compute_post_kl:
            with timed("compute_post_kl", metrics):
                post_lp = self.trainer.logprobs(data, "policy")  # all ranks
                if is_main:
                    metrics.update(await compute_post_kl(data, PrecomputedLogprobs(data, post_lp)))

        # Safe point 4: the step is complete; save it and stop if preempted (also if the vLLM
        # sync fails because vLLM was preempted with us).
        preempt = self.comm.agree(self.flag.requested)
        if not preempt:  # all ranks: save_adapter is collective under FSDP
            with timed("sync_adapter", metrics):
                preempt = not self._sync_adapter(step)
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

    def _abandon(self, i_batch: int, rnd_before: dict | None) -> bool:
        """All ranks: drop the in-progress step, checkpoint the last completed one (``i_batch``
        steps done; RND state from before this step's shaping), and report preemption."""
        logger.info(f"Preemption requested during step {i_batch}: abandoning it")
        self._save(i_batch, rnd_before)
        return False


async def run(cfg: LeanRLConfig, *, transport=None) -> None:
    """Run (or resume) training. ``transport`` is a test hook for the vLLM HTTP client."""
    await _Loop(cfg, transport=transport).run()
