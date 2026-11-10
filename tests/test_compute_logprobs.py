"""compute_logprobs_async on LocalSamplingClient, against a mocked vLLM server.

Response shape derived from vLLM 0.20.2 (entrypoints/openai/completion/protocol.py
`CompletionResponseChoice.prompt_logprobs: list[dict[int, Logprob] | None]`, where
`Logprob` is a dataclass {logprob, rank, decoded_token}; JSON keys are token-id strings;
entry 0 is None; with prompt_logprobs=0 the dict holds the actual token (plus the
top-ranked token when it differs); -inf is clamped to -9999.0 by the server).
"""
import asyncio
import json

import httpx
import pytest
import tinker

from local_backend.sampling_client import LocalSamplingClient


def _lp(logprob, rank=1, tok="x"):
    return {"logprob": logprob, "rank": rank, "decoded_token": tok}


def _client(prompt_logprobs, captured, adapter_name=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"index": 0, "text": "a", "finish_reason": "length",
                         "prompt_logprobs": prompt_logprobs}],
        })

    return LocalSamplingClient(
        vllm_urls=["http://vllm:8000"], base_model_name="base/model",
        adapter_name=adapter_name, transport=httpx.MockTransport(handler),
    )


def test_compute_logprobs_alignment():
    captured = []
    # Position 1: actual token 17 has logprob -1.5, but a higher-ranked id 4 is also present.
    # Position 2: actual token 9 is the only entry.
    resp = [None,
            {"4": _lp(-0.1, rank=1), "17": _lp(-1.5, rank=3)},
            {"9": _lp(-0.25)}]
    client = _client(resp, captured)
    out = asyncio.run(client.compute_logprobs_async(tinker.ModelInput.from_ints([5, 17, 9])))
    assert out == [None, -1.5, -0.25]
    body = captured[0]
    assert body["prompt"] == [5, 17, 9]
    assert body["max_tokens"] == 1
    assert body["prompt_logprobs"] == 0
    assert body["model"] == "base/model"


def test_compute_logprobs_uses_adapter_name():
    captured = []
    client = _client([None, {"2": _lp(-2.0)}], captured, adapter_name="kl_ref")
    out = client.compute_logprobs(tinker.ModelInput.from_ints([1, 2]))
    assert out == [None, -2.0]
    assert captured[0]["model"] == "kl_ref"


def test_compute_logprobs_length_mismatch_raises():
    captured = []
    client = _client([None, {"17": _lp(-1.5)}], captured)  # one fewer than 3 tokens
    with pytest.raises(RuntimeError):
        asyncio.run(client.compute_logprobs_async(tinker.ModelInput.from_ints([5, 17, 9])))


def test_compute_logprobs_missing_actual_token_raises():
    captured = []
    client = _client([None, {"99": _lp(-1.5)}], captured)
    with pytest.raises(RuntimeError):
        asyncio.run(client.compute_logprobs_async(tinker.ModelInput.from_ints([5, 17])))
