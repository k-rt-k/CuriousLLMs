import pytest


def test_local_backend_returns_local_client(monkeypatch):
    from local_backend.backend import make_service_client
    from local_backend import LocalServiceClient

    c = make_service_client("local", base_url="http://x:1", log_path="/tmp/l", model_name="m")
    assert isinstance(c, LocalServiceClient) and c.vllm_urls == ["http://x:1"]


def test_tinker_backend_returns_tinker_client(monkeypatch):
    import tinker
    from local_backend.backend import make_service_client

    monkeypatch.setattr(tinker, "ServiceClient", lambda base_url=None: ("tinker", base_url))
    assert make_service_client("tinker", base_url=None, log_path=None, model_name="m") == ("tinker", None)


def test_init_from_adapter_rejected_on_tinker():
    from types import SimpleNamespace
    from curiosity_train import validate_backend_config

    # validate_backend_config reads only these attributes, so a namespace stands in for Config
    cfg = SimpleNamespace(backend="tinker", init_from_adapter="/a", kl_reference="base")
    with pytest.raises(ValueError, match="init_from_adapter"):
        validate_backend_config(cfg)


def test_kl_reference_init_requires_adapter_and_local():
    from types import SimpleNamespace
    from curiosity_train import validate_backend_config

    with pytest.raises(ValueError, match="kl_reference"):
        validate_backend_config(SimpleNamespace(backend="local", init_from_adapter=None, kl_reference="init"))
    with pytest.raises(ValueError, match="kl_reference"):
        validate_backend_config(SimpleNamespace(backend="tinker", init_from_adapter=None, kl_reference="init"))
    validate_backend_config(SimpleNamespace(backend="local", init_from_adapter="/a", kl_reference="init"))
    validate_backend_config(SimpleNamespace(backend="local", init_from_adapter=None, kl_reference="base"))


class _FakeService:
    def __init__(self):
        self.calls = []

    def create_sampling_client(self, **kw):
        self.calls.append(kw)
        return ("client", kw)


def test_make_kl_reference_client_base_and_init():
    from types import SimpleNamespace
    from curiosity_train import make_kl_reference_client

    svc = _FakeService()
    cfg = SimpleNamespace(model_name="m", kl_reference="base", init_from_adapter="/a")
    make_kl_reference_client(svc, cfg)
    assert svc.calls[-1] == {"base_model": "m"}

    cfg = SimpleNamespace(model_name="m", kl_reference="init", init_from_adapter="/a")
    make_kl_reference_client(svc, cfg)
    assert svc.calls[-1] == {"model_path": "/a", "base_model": "m", "adapter_name": "kl_ref"}
