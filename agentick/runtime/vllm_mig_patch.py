"""Compatibility patch for vLLM when SLURM exposes a CUDA MIG UUID."""

from __future__ import annotations

import os
from typing import Any


def _patch_platform_class(
    platform_cls: type,
    pynvml: Any,
    device_capability_cls: type,
    visible_device_ids: tuple[str, ...],
) -> None:
    """Teach vLLM's NVML platform helpers to query CUDA MIG UUID handles."""

    def visible_id(device_id: int) -> str:
        return visible_device_ids[device_id]

    def query(device_id: int, fn, *, use_parent: bool = False):
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByUUID(visible_id(device_id))
            if use_parent:
                handle = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
            return fn(handle)
        finally:
            pynvml.nvmlShutdown()

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        del cls
        return visible_id(device_id)

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        del cls
        major, minor = query(
            device_id,
            pynvml.nvmlDeviceGetCudaComputeCapability,
            use_parent=True,
        )
        return device_capability_cls(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0):
        del cls
        return query(device_id, pynvml.nvmlDeviceGetName, use_parent=True)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0):
        del cls
        return query(device_id, pynvml.nvmlDeviceGetUUID)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0):
        del cls
        memory = query(device_id, pynvml.nvmlDeviceGetMemoryInfo)
        return int(memory.total)

    @classmethod
    def is_fully_connected(cls, physical_device_ids):
        del cls
        # A single MIG device has no peer-connectivity requirement. Tensor
        # parallelism across multiple MIG instances is outside this runtime.
        return len(physical_device_ids) <= 1

    platform_cls.device_id_to_physical_device_id = device_id_to_physical_device_id
    platform_cls.get_device_capability = get_device_capability
    platform_cls.get_device_name = get_device_name
    platform_cls.get_device_uuid = get_device_uuid
    platform_cls.get_device_total_memory = get_device_total_memory
    platform_cls.is_fully_connected = is_fully_connected


def install_vllm_mig_uuid_support() -> bool:
    """Install the patch when all visible CUDA devices are MIG UUIDs."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_device_ids = tuple(item.strip() for item in visible.split(",") if item.strip())
    if not visible_device_ids or not all(
        item.startswith("MIG-") for item in visible_device_ids
    ):
        return False

    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return False

    from vllm.platforms.cuda import pynvml
    from vllm.platforms.interface import DeviceCapability

    _patch_platform_class(
        type(current_platform),
        pynvml,
        DeviceCapability,
        visible_device_ids,
    )
    return True
