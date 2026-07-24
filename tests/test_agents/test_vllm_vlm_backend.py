"""Focused tests for the vLLM vision-language backend."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from agentick.agents.backends.vllm_vlm import VLLMVLMBackend
from agentick.runtime.vllm_mig_patch import _patch_platform_class


def test_vllm_vlm_backend_constructs_with_eager_disabled():
    backend = VLLMVLMBackend(enforce_eager=False)

    assert backend.enforce_eager is False


def test_vllm_vlm_backend_forwards_eager_when_enabled(monkeypatch):
    captured_kwargs = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )

    backend = VLLMVLMBackend(enforce_eager=True)
    backend._ensure_loaded()

    assert captured_kwargs["enforce_eager"] is True
    assert captured_kwargs["attention_backend"] == "TRITON_ATTN"


def test_vllm_mig_patch_preserves_uuid_and_queries_slice_memory():
    calls = []

    class FakePlatform:
        pass

    class FakeCapability:
        def __init__(self, major, minor):
            self.major = major
            self.minor = minor

    class FakePynvml:
        @staticmethod
        def nvmlInit():  # noqa: N802
            calls.append("init")

        @staticmethod
        def nvmlShutdown():  # noqa: N802
            calls.append("shutdown")

        @staticmethod
        def nvmlDeviceGetHandleByUUID(uuid):  # noqa: N802
            calls.append(("uuid", uuid))
            return f"handle:{uuid}"

        @staticmethod
        def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle):  # noqa: N802
            assert handle == "handle:MIG-test"
            calls.append(("parent", handle))
            return f"parent:{handle}"

        @staticmethod
        def nvmlDeviceGetCudaComputeCapability(handle):  # noqa: N802
            assert handle == "parent:handle:MIG-test"
            return 9, 0

        @staticmethod
        def nvmlDeviceGetName(handle):  # noqa: N802
            assert handle == "parent:handle:MIG-test"
            return "NVIDIA H100 80GB HBM3"

        @staticmethod
        def nvmlDeviceGetUUID(handle):  # noqa: N802
            assert handle == "handle:MIG-test"
            return "MIG-test"

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):  # noqa: N802
            assert handle == "handle:MIG-test"
            return SimpleNamespace(total=40 * 1024**3)

    _patch_platform_class(
        FakePlatform,
        FakePynvml,
        FakeCapability,
        ("MIG-test",),
    )

    assert FakePlatform.device_id_to_physical_device_id(0) == "MIG-test"
    assert FakePlatform.get_device_total_memory() == 40 * 1024**3
    capability = FakePlatform.get_device_capability()
    assert (capability.major, capability.minor) == (9, 0)
    assert FakePlatform.get_device_name() == "NVIDIA H100 80GB HBM3"
    assert FakePlatform.get_device_uuid() == "MIG-test"
    assert FakePlatform.is_fully_connected(["MIG-test"]) is True
    assert calls.count(("uuid", "MIG-test")) == 4
    assert calls.count(("parent", "handle:MIG-test")) == 2
