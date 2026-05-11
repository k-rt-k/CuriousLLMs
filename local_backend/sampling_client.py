"""
LocalSamplingClient — a duck-typed replacement for `tinker.SamplingClient`
backed by a running vLLM OpenAI-compatible server.

Tinker call shape we must implement (from
tinker_cookbook.completers.TinkerTokenCompleter:64 and math_evaluation.py):

    response: tinker.SampleResponse = await client.sample_async(
        prompt=tinker.ModelInput,
        num_samples=int,
        sampling_params=tinker.SamplingParams(stop, max_tokens, temperature, top_p, top_k, seed),
    )
    # then:
    response.sequences[i].tokens     # list[int]
    response.sequences[i].logprobs   # list[float] | None
    response.sequences[i].stop_reason

We send vLLM `prompt_token_ids` (not text) so the response token ids align
exactly with what the trainer will score later.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import httpx
import tinker

logger = logging.getLogger(__name__)


def _normalize_stop(stop) -> tuple[Optional[list[str]], Optional[list[int]]]:
    """tinker.SamplingParams.stop is `str | Sequence[str] | Sequence[int] | None`.

    vLLM's OpenAI server accepts:
        stop:           list[str]    (text)
        stop_token_ids: list[int]
    """
    if stop is None:
        return None, None
    if isinstance(stop, str):
        return [stop], None
    seq = list(stop)
    if not seq:
        return None, None
    if all(isinstance(x, int) for x in seq):
        return None, seq
    return [str(x) for x in seq], None


def _model_input_to_token_ids(model_input) -> list[int]:
    """Accept tinker.ModelInput or a raw list[int]."""
    if isinstance(model_input, tinker.ModelInput):
        return model_input.to_ints()
    if isinstance(model_input, (list, tuple)) and all(isinstance(x, int) for x in model_input):
        return list(model_input)
    raise TypeError(f"Unsupported prompt type: {type(model_input)}")


class LocalSamplingClient:
    """
    Sample from vLLM. The adapter served is selected by `adapter_name`:
      - None or self.base_model_name -> serve the base model with no LoRA
      - else                          -> serve the named LoRA adapter
    """

    def __init__(
        self,
        *,
        vllm_url: str,
        base_model_name: str,
        adapter_name: Optional[str] = None,
        tokenizer=None,
        request_timeout: float = 600.0,
    ):
        self.vllm_url = vllm_url.rstrip("/")
        self.base_model_name = base_model_name
        self.adapter_name = adapter_name
        self.tokenizer = tokenizer  # optional; only needed for compute_logprobs
        self._client = httpx.AsyncClient(timeout=request_timeout)

    def _model_field(self) -> str:
        return self.adapter_name or self.base_model_name

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError(
                "LocalSamplingClient was created without a tokenizer. "
                "Pass `tokenizer=...` if you need .get_tokenizer()."
            )
        return self.tokenizer

    async def sample_async(
        self,
        prompt,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
    ) -> tinker.SampleResponse:
        prompt_ids = _model_input_to_token_ids(prompt)
        stop_text, stop_ids = _normalize_stop(sampling_params.stop)

        body = {
            "model": self._model_field(),
            "prompt": prompt_ids,  # vLLM accepts list[int] as prompt_token_ids
            "n": num_samples,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_tokens,
            "logprobs": 1,
        }
        if sampling_params.top_k is not None and sampling_params.top_k > 0:
            body["top_k"] = sampling_params.top_k
        if sampling_params.seed is not None:
            body["seed"] = sampling_params.seed
        if stop_text is not None:
            body["stop"] = stop_text
        if stop_ids is not None:
            body["stop_token_ids"] = stop_ids

        r = await self._client.post(self.vllm_url + "/v1/completions", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"vLLM /v1/completions -> {r.status_code}: {r.text[:500]}")
        data = r.json()

        return _to_sample_response(data, prompt_ids)

    # synchronous variant for parity (some tinker_cookbook code paths call .sample())
    def sample(self, prompt, num_samples: int, sampling_params: tinker.SamplingParams) -> tinker.SampleResponse:
        import asyncio
        return asyncio.run(self.sample_async(prompt, num_samples, sampling_params))

    async def compute_logprobs_async(self, *args, **kwargs):
        raise NotImplementedError(
            "LocalSamplingClient.compute_logprobs_async is not implemented. "
            "Use LocalTrainingClient.forward_async for teacher-forced logprobs."
        )

    def compute_logprobs(self, *args, **kwargs):
        raise NotImplementedError(
            "LocalSamplingClient.compute_logprobs is not implemented."
        )

    async def aclose(self):
        await self._client.aclose()


def _to_sample_response(data: dict, prompt_ids: Sequence[int]) -> tinker.SampleResponse:
    """
    vLLM completion response -> tinker.SampleResponse.

    OpenAI completions schema:
      choices[i] = {
        "text": str,
        "finish_reason": "stop" | "length",
        "logprobs": {
          "tokens": [str],          # detokenized response tokens (may not be aligned to single-id boundaries)
          "token_logprobs": [float],
          "top_logprobs": [...],
          "text_offset": [int],
        }
      }

    We need response *token ids*, not text. vLLM's OpenAI server includes the
    sampled token ids inside `logprobs.tokens` as the textual form of each
    token. We rely on the modern vLLM extension that adds `token_ids` to the
    logprobs object; if it's absent we fall back to re-tokenizing the text
    (less reliable — surface a warning).
    """
    sequences = []
    for choice in data["choices"]:
        lp = choice.get("logprobs") or {}
        token_ids = lp.get("token_ids")
        token_logprobs = lp.get("token_logprobs") or []
        if token_ids is None:
            # vLLM newer than 0.6 also returns "tokens" as ["token_id:NNNN"] when
            # the request includes prompt_token_ids. Try parsing.
            raw_tokens = lp.get("tokens") or []
            token_ids = _parse_vllm_token_ids(raw_tokens)
        # Drop the leading None on the first token_logprob (vLLM convention)
        if token_logprobs and token_logprobs[0] is None:
            token_logprobs = token_logprobs[1:]
        # Align lengths defensively
        n = min(len(token_ids), len(token_logprobs)) if token_logprobs else len(token_ids)
        if token_logprobs:
            token_ids = token_ids[:n]
            token_logprobs = token_logprobs[:n]

        finish = choice.get("finish_reason", "stop")
        stop_reason = "length" if finish == "length" else "stop"

        sequences.append(
            tinker.SampledSequence(
                stop_reason=stop_reason,
                tokens=list(token_ids),
                logprobs=list(token_logprobs) if token_logprobs else None,
            )
        )
    return tinker.SampleResponse(sequences=sequences)


def _parse_vllm_token_ids(raw_tokens: list[str]) -> list[int]:
    """
    vLLM's logprobs.tokens often contains entries shaped like 'token_id:NNNN'
    or just the textual form. Best-effort parse to integer ids.
    """
    out: list[int] = []
    for t in raw_tokens:
        if isinstance(t, int):
            out.append(t)
            continue
        if isinstance(t, str) and t.startswith("token_id:"):
            try:
                out.append(int(t.split(":", 1)[1]))
                continue
            except ValueError:
                pass
        # If we can't parse, abort — the caller will see a length mismatch
        raise RuntimeError(
            f"Cannot extract token_id from vLLM logprobs entry {t!r}. "
            "Upgrade vLLM to a version that returns token_ids in the completions response."
        )
    return out
