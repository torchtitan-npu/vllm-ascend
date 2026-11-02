"""Intentional negative payload for a non-mergeable Blue acceptance PR.

The trusted controller must prepare this PR's merge source successfully, then
report an actual NPU execution failure and verified cleanup. Never merge this
acceptance branch into the production controller.
"""

PAYLOAD_API_VERSION = 1


def all_reduce(tensor):
    raise RuntimeError("EXPECTED_BLUE_ACCEPTANCE_PAYLOAD_FAILURE_20261102")
