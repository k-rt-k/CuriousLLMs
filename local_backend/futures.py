"""
Tinker uses a future pattern:

    fwd_bwd_future = await client.forward_backward_async(...)
    result = await fwd_bwd_future.result_async()

We need to mimic that two-stage await even though our backend is synchronous.
`LocalFuture(result)` wraps a value; `.result_async()` returns it.
"""

from typing import Generic, TypeVar

T = TypeVar("T")


class LocalFuture(Generic[T]):
    __slots__ = ("_result",)

    def __init__(self, result: T):
        self._result = result

    async def result_async(self) -> T:
        return self._result

    def result(self) -> T:
        return self._result


class SavedPath:
    """
    Return value of `save_state_async(name).result_async()` and
    `save_weights_for_sampler_async(name).result_async()`.
    tinker_cookbook.checkpoint_utils reads `.path`.
    """

    __slots__ = ("path",)

    def __init__(self, path: str):
        self.path = path
