"""Install Agentick's vLLM MIG UUID compatibility patch at process startup."""

from agentick.runtime.vllm_mig_patch import install_vllm_mig_uuid_support

if install_vllm_mig_uuid_support():
    print("[agentick] enabled vLLM MIG UUID support", flush=True)
