"""Tested source payload for the Lite Actions Blue A3 HCCL smoke.

The immutable Worker runner owns topology, timeout, expected sums and evidence.
This function is loaded only inside the leased NPU container, never in Prepare.
"""

PAYLOAD_API_VERSION = 1


def all_reduce(tensor):
    import torch.distributed as dist

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
