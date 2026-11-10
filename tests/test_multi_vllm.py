"""Multiple vLLM servers: load-balanced sampling, fan-out LoRA sync, URL parsing."""
import asyncio
import collections
import json

import httpx
import pytest
import tinker

from local_backend.sampling_client import LocalSamplingClient
from local_backend import weight_sync
from local_backend.weight_sync import sync_lora_to_vllm


def _ok_completion():
    return {"choices": [{"index": 0, "text": "a", "finish_reason": "length",
                         "logprobs": {"tokens": ["token_id:5"], "token_logprobs": [-0.5]}}]}


def test_requests_spread_across_servers():
    hosts = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        await asyncio.sleep(0.01)  # keep requests in flight so the balancer sees load
        return httpx.Response(200, json=_ok_completion())

    client = LocalSamplingClient(
        vllm_urls=["http://a:1", "http://b:1"], base_model_name="m",
        transport=httpx.MockTransport(handler),
    )
    params = tinker.SamplingParams(max_tokens=1, temperature=1.0)

    async def go():
        await asyncio.gather(*[
            client.sample_async(tinker.ModelInput.from_ints([1, 2]), 1, params) for _ in range(8)
        ])

    asyncio.run(go())
    assert collections.Counter(hosts) == {"a": 4, "b": 4}
    assert all(v == 0 for v in client._in_flight.values())


def test_in_flight_counter_released_on_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    client = LocalSamplingClient(
        vllm_urls=["http://a:1"], base_model_name="m", transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError):
        asyncio.run(client.compute_logprobs_async(tinker.ModelInput.from_ints([1, 2])))
    assert client._in_flight == {"http://a:1": 0}


def test_sampling_client_requires_urls():
    with pytest.raises(RuntimeError, match="vllm_urls"):
        LocalSamplingClient(vllm_urls=[], base_model_name="m")


def _recording_transport(fail_host=None):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.host, request.url.path, json.loads(request.content)))
        if request.url.host == fail_host and request.url.path == "/v1/load_lora_adapter":
            return httpx.Response(500, text="kaboom")
        return httpx.Response(200, json={})

    return calls, handler


def _patch_post(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def post(url, json=None, timeout=None):
        with httpx.Client(transport=transport) as c:
            return c.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(weight_sync.httpx, "post", post)


def test_lora_sync_hits_every_server(monkeypatch):
    calls, handler = _recording_transport()
    _patch_post(monkeypatch, handler)
    sync_lora_to_vllm("/ad/new", "new", ["http://a:1", "http://b:1"], prev_name="old")
    loads = {h for h, p, _ in calls if p == "/v1/load_lora_adapter"}
    unloads = [(h, b) for h, p, b in calls if p == "/v1/unload_lora_adapter"]
    assert loads == {"a", "b"}
    assert sorted(h for h, _ in unloads) == ["a", "b"]
    assert all(b == {"lora_name": "old"} for _, b in unloads)


def test_lora_sync_partial_failure_raises(monkeypatch):
    calls, handler = _recording_transport(fail_host="b")
    _patch_post(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="b:1"):
        sync_lora_to_vllm("/ad/new", "new", ["http://a:1", "http://b:1"])
    # the healthy server was still updated
    assert any(h == "a" and p == "/v1/load_lora_adapter" for h, p, _ in calls)


def test_sync_with_no_urls_raises():
    with pytest.raises(RuntimeError, match="vllm_urls"):
        sync_lora_to_vllm("/ad", "n", [])


def test_env_vllm_urls_parsing(monkeypatch):
    from local_backend import LocalServiceClient

    monkeypatch.setenv("VLLM_URLS", "http://a:1, http://b:2")
    monkeypatch.setenv("VLLM_URL", "http://ignored:9")
    assert LocalServiceClient().vllm_urls == ["http://a:1", "http://b:2"]
    monkeypatch.delenv("VLLM_URLS")
    assert LocalServiceClient().vllm_urls == ["http://ignored:9"]
    assert LocalServiceClient(vllm_urls="http://x:1/").vllm_urls == ["http://x:1"]


def test_make_service_client_comma_separated(monkeypatch):
    from local_backend.backend import make_service_client

    c = make_service_client("local", base_url="http://a:1,http://b:1", log_path=None, model_name="m")
    assert c.vllm_urls == ["http://a:1", "http://b:1"]
    monkeypatch.setenv("VLLM_URLS", "http://c:1,http://d:1")
    c = make_service_client("local", base_url=None, log_path=None, model_name="m")
    assert c.vllm_urls == ["http://c:1", "http://d:1"]


def test_create_sampling_client_loads_on_every_server(monkeypatch):
    from local_backend import LocalServiceClient, service_client

    seen = []
    monkeypatch.setattr(service_client, "load_lora", lambda urls, name, d: seen.append((urls, name, d)))
    svc = LocalServiceClient(vllm_urls=["http://a:1", "http://b:1"])
    sc = svc.create_sampling_client(model_path="/x/kl", base_model="m", adapter_name="kl_ref")
    assert seen == [(["http://a:1", "http://b:1"], "kl_ref", "/x/kl")]
    assert sc.vllm_urls == ["http://a:1", "http://b:1"]


def test_training_client_accepts_str_and_empty():
    from local_backend.training_client import _normalize_urls

    assert _normalize_urls("http://a:1/") == ["http://a:1"]
    assert _normalize_urls([]) == []
