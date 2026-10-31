# Lite Actions A5 PR integration verification

This non-production file exists to create a real fork-originated Pull Request to `torchtitan-npu/vllm-ascend:main`.

Expected: the single `Dispatcher-Smoke` workflow creates `dispatch_smoke_a5_8p`, the Yellow `a5_shared_yellow` worker claims the request, runs HCCL all-reduce and posts a schema-3 result. The `lite-a5-32p` label opts into a 32P case; do not label this initial 8P validation PR.

No application code or upstream community behavior is altered by this validation change.
