"""Trusted Blue smoke specification; no model imports or device access."""

BLUE_CASES = {
    "vllm_worker_os_blue": (0, 0, 90),
    "dispatch_smoke_a3_8p": (1, 8, 180),
    "dispatch_smoke_a3_16p": (2, 8, 240),
}
BLUE_SMOKE_SUITE = tuple(BLUE_CASES)
BLUE_DEFAULTS = dict(worker="blue", target_pool="a3_shared_blue", docker="vllm_hccl_smoke")
