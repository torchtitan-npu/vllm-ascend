#!/usr/bin/env python3
"""Trusted actor's PR SHA A5 collective smoke. Runs only inside authorized NPU container."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-id", required=True, choices=("dispatch_smoke_a5_8p", "dispatch_smoke_a5_32p"))
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    expected = 8 if a.test_id.endswith("_8p") else 32
    if world_size != expected:
        raise RuntimeError(f"invalid world size {world_size}; expected {expected}")
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        tensor = torch.tensor([float(rank)], device=f"npu:{local_rank}")
        dist.all_reduce(tensor)
        actual = tensor.cpu().item()
        target = world_size * (world_size - 1) // 2
        if actual != target:
            raise RuntimeError(f"all_reduce mismatch {actual} != {target}")
        dist.barrier()
        if rank == 0:
            path = Path(a.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            evidence = {"test_id": a.test_id, "status": "PASS", "world_size": world_size, "all_reduce_sum": target}
            (path / "a5-smoke-evidence.json").write_text(json.dumps(evidence))
            print("DISPATCH_SMOKE_PASS " + json.dumps(evidence), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
