"""
Format-SFT dataset builder.

Loads a HuggingFace dataset of (problem, solution) pairs, applies the model's
chat template, builds response-masked `tinker.Datum` objects that the local
NLL training path consumes.

This is the data side of the cold-start format SFT stage (DeepSeek-R1 style):
SFT to a fixed output format (e.g. `<think>...</think> \\boxed{answer}`) before
RL kicks in, so RL doesn't have to learn formatting from a random init.
"""

from __future__ import annotations

import logging
import random
from typing import Iterator, Optional

import tinker
import torch
from transformers import AutoTokenizer

# tinker-cookbook submodule helper that builds a response-masked Datum.
# Path: tinker-cookbook/tinker_cookbook/supervised/common.py:29
from tinker_cookbook.supervised.common import datum_from_tokens_weights

logger = logging.getLogger(__name__)


def _format_prompt_and_full(
    tokenizer,
    problem: str,
    solution: str,
) -> tuple[list[int], list[int]] | None:
    """Return (prompt_token_ids, full_token_ids) for one (problem, solution) row.

    Uses `apply_chat_template` when the tokenizer has one (most instruct models),
    otherwise falls back to a "User: ... Assistant: ..." plain-text wrapper.

    `solution` is appended verbatim — the upstream dataset is expected to
    already contain the target format (e.g. `\\boxed{answer}`).

    Returns ``None`` if prompt tokens are not a prefix of full tokens — a BPE
    boundary hazard when encoding ``prompt + solution`` as a single string can
    produce different token boundaries than encoding each piece separately.
    Affected rows are dropped by the caller rather than risking a misaligned
    response mask.
    """
    eos = tokenizer.eos_token or ""
    try:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + solution + eos
    except (AttributeError, NotImplementedError, ValueError):
        prompt_text = f"User: {problem}\nAssistant: "
        full_text = prompt_text + solution + eos

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return None
    return prompt_ids, full_ids


def build_format_data(
    *,
    model_name: str,
    dataset_name: str,
    dataset_split: str = "train",
    dataset_config: Optional[str] = None,
    problem_column: str = "problem",
    solution_column: str = "solution",
    max_seq_len: int = 4096,
    max_rows: Optional[int] = None,
    seed: int = 0,
) -> list[tinker.Datum]:
    """Load the HF dataset and build a list of response-masked Datums.

    Returns a flat list ordered by the (shuffled) source dataset. Caller
    typically wraps it with `iter_batches` for training.
    """
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pass cache_dir explicitly so we don't get bitten by datasets' offline
    # fallback scanning unwritable shared cache locations on compute nodes.
    import os as _os
    cache_dir = _os.environ.get("HF_DATASETS_CACHE")
    if dataset_config is not None:
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split, cache_dir=cache_dir)
    else:
        ds = load_dataset(dataset_name, split=dataset_split, cache_dir=cache_dir)
    ds = ds.shuffle(seed=seed)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    datums: list[tinker.Datum] = []
    skipped_long = 0
    skipped_empty = 0
    skipped_misalign = 0
    for row in ds:
        problem = row.get(problem_column)
        solution = row.get(solution_column)
        if not problem or not solution:
            skipped_empty += 1
            continue

        pair = _format_prompt_and_full(tokenizer, problem, solution)
        if pair is None:
            skipped_misalign += 1
            continue
        prompt_ids, full_ids = pair
        if len(full_ids) <= len(prompt_ids):
            skipped_empty += 1
            continue
        if len(full_ids) > max_seq_len:
            skipped_long += 1
            continue

        weights_list = [0.0] * len(prompt_ids) + [1.0] * (len(full_ids) - len(prompt_ids))
        tokens_t = torch.tensor(full_ids, dtype=torch.long)
        weights_t = torch.tensor(weights_list, dtype=torch.float32)
        datums.append(datum_from_tokens_weights(tokens_t, weights_t, max_length=max_seq_len))

    logger.info(
        "format_dataset: built %d Datums from %s/%s "
        "(skipped: %d empty, %d too long, %d tokenizer-misaligned)",
        len(datums), dataset_name, dataset_split,
        skipped_empty, skipped_long, skipped_misalign,
    )
    if not datums:
        raise RuntimeError(
            f"No usable rows in {dataset_name}/{dataset_split}. "
            f"Check problem_column={problem_column!r}, solution_column={solution_column!r}, "
            f"and max_seq_len={max_seq_len}."
        )
    return datums


def iter_batches(
    datums: list[tinker.Datum],
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[list[tinker.Datum]]:
    """Yield contiguous micro-batches of size `batch_size`, looping forever."""
    rng = random.Random(seed)
    n = len(datums)
    while True:
        order = list(range(n))
        if shuffle:
            rng.shuffle(order)
        for i in range(0, n, batch_size):
            yield [datums[j] for j in order[i:i + batch_size]]
