"""
Local SLURM-native backend that replaces Tinker's network clients.

We **keep** the `tinker` Python package installed — but only for its data types
(`tinker.ModelInput`, `tinker.Datum`, `tinker.SamplingParams`, `tinker.SampleResponse`,
`tinker.SampledSequence`, `tinker.AdamParams`, `tinker.TensorData`, `tinker.ForwardBackwardOutput`).
Tinker-cookbook constructs these as plain data carriers; our local clients consume
them and produce them.

We **replace** the three network-talking classes:

    tinker.ServiceClient  ->  local_backend.LocalServiceClient
    tinker.TrainingClient ->  local_backend.LocalTrainingClient   (HF + PEFT + AdamW)
    tinker.SamplingClient ->  local_backend.LocalSamplingClient   (vLLM HTTP)

This is a duck-typed adapter: tinker-cookbook never does isinstance checks on the
client objects, only calls methods on them. As long as our objects expose the
same method names with the same async signatures and return Pydantic-shaped
objects, the rest of tinker-cookbook (`do_group_rollout`, `checkpoint_utils`,
`data_processing`, `metric_util`, `eval.evaluators`, `completers.TinkerTokenCompleter`)
works unchanged.

Entry point:

    from local_backend import LocalServiceClient
    service = LocalServiceClient(vllm_url=..., model_name=...)
    training_client = await service.create_lora_training_client_async(
        base_model=model_name, rank=32
    )
"""

from local_backend.service_client import LocalServiceClient
from local_backend.training_client import LocalTrainingClient
from local_backend.sampling_client import LocalSamplingClient

__all__ = [
    "LocalServiceClient",
    "LocalTrainingClient",
    "LocalSamplingClient",
]
