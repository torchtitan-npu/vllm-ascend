#!/usr/bin/env python3
"""Trusted schema-3 plan builder. Never run or import fork Python in control jobs."""

import hashlib
import json
import os
import re
from pathlib import Path

from blue_catalog import BLUE_CASES, BLUE_DEFAULTS, BLUE_SMOKE_SUITE

REPO = "cann-framework/vllm-ascend"
WORKFLOW = "lite_actions_dispatcher_smoke.yml"
A5_CASES = {"dispatch_smoke_a5_8p": (1, 8, 120), "dispatch_smoke_a5_32p": (4, 8, 240)}
CASES = {**A5_CASES, **BLUE_CASES}
DEFAULTS = dict(worker="yellow", target_pool="a5_shared_yellow", docker="vllm-kimi-wangjinyin")


def selectors(event, payload, value):
    if event == "pull_request_target":
        pr = payload["pull_request"]
        base, head = pr["base"], pr["head"]
        target = base["ref"]
        if (
            base["repo"]["full_name"] != REPO
            or not (target == "main" or target.startswith("releases/") or target.endswith("-dev"))
            or head["repo"]["full_name"] != "depeng1994/vllm-ascend"
            or pr["user"]["login"] != "depeng1994"
            or pr.get("draft")
            or pr["state"] != "open"
        ):
            raise ValueError("PR is not on the trusted admission allowlist")
        sha = head["sha"]
        ids = ["dispatch_smoke_a5_8p"]
        labels = {x["name"] for x in pr.get("labels", [])}
        if "lite-a3-blue" in labels:
            if target != "main":
                raise ValueError("Blue merge tests only target main")
            ids = list(BLUE_SMOKE_SUITE)
        elif "lite-a5-32p" in labels:
            ids.append("dispatch_smoke_a5_32p")
    elif event == "workflow_dispatch":
        if os.environ.get("GITHUB_REF") != "refs/heads/main":
            raise ValueError("manual run only supported from main")
        sha = os.environ["GITHUB_SHA"]
        if not value:
            ids = ["dispatch_smoke_a5_8p"]
        else:
            values = json.loads(value)
            if not isinstance(values, list) or not 1 <= len(values) <= 3:
                raise ValueError("invalid smoke test selection")
            if any(not isinstance(x, dict) or set(x) != {"test_id"} for x in values):
                raise ValueError("only fixed test_id selectors supported")
            ids = [x["test_id"] for x in values]
    else:
        raise ValueError("unsupported GitHub event")
    if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{40}", sha):
        raise ValueError("invalid source SHA")
    if not ids or len(ids) != len(set(ids)) or any(x not in CASES for x in ids):
        raise ValueError("unknown/duplicate smoke case")
    if any(x in BLUE_CASES for x in ids) and any(x in A5_CASES for x in ids):
        raise ValueError("Blue and Yellow cases cannot share a native source plan")
    return sha, ids


def create_plan(event, payload, value, run_id, attempt):
    sha, ids = selectors(event, payload, value)
    defaults = BLUE_DEFAULTS if ids[0] in BLUE_CASES else DEFAULTS
    items = []
    for i, test_id in enumerate(ids):
        nodes, ngpu, timeout = CASES[test_id]
        items.append(
            {
                "case_id": f"case_{i:03d}_{test_id}",
                "order": i,
                "test_id": test_id,
                "label": "vLLM Blue worker OS probe"
                if nodes == 0
                else f"{'A3' if test_id in BLUE_CASES else 'A5'} HCCL distributed smoke {nodes * ngpu}P",
                "kind": "worker_probe" if nodes == 0 else "npu_smoke",
                "worker": defaults["worker"],
                "target_pool": defaults["target_pool"] if nodes else None,
                "resources": {"nodes": nodes, "devices_per_node": ngpu},
                "docker": defaults["docker"] if nodes else None,
                "execution_timeout_seconds": timeout,
            }
        )
    plan = {
        "schema": 3,
        "repo": REPO,
        "workflow": WORKFLOW,
        "run_id": run_id,
        "attempt": attempt,
        "sha": sha,
        "defaults": defaults,
        "cases": items,
    }
    plan["plan_digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return plan


def main():
    if os.environ["GITHUB_REPOSITORY"] != REPO:
        raise ValueError("untrusted repository")
    payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    effective_event = os.environ["GITHUB_EVENT_NAME"]
    source_pr = os.environ.get("CI_SOURCE_PR", "")
    if source_pr:
        if effective_event != "workflow_dispatch" or not re.fullmatch(r"[1-9][0-9]{0,8}", source_pr):
            raise ValueError("invalid native PR recheck")
        from native_source import current_pr

        pr, head, base = current_pr(int(source_pr))
        if (head, base) != (
            os.environ.get("CI_EXPECTED_HEAD"),
            os.environ.get("CI_EXPECTED_BASE"),
        ):
            raise ValueError("STALE_RECHECK_INTENT")
        if not any(x["name"] == "lite-a3-blue" for x in pr.get("labels", [])):
            raise ValueError("Blue recheck requires opt-in label")
        payload = {"pull_request": pr}
        effective_event = "pull_request_target"
    plan = create_plan(
        effective_event,
        payload,
        os.environ.get("CI_TEST_CASES", ""),
        int(os.environ["GITHUB_RUN_ID"]),
        int(os.environ["GITHUB_RUN_ATTEMPT"]),
    )
    out = Path(".ci-request")
    out.mkdir(exist_ok=True)
    native = plan["cases"][0]["test_id"] in BLUE_CASES
    if native:
        from native_source import prepare_source

        source = prepare_source(plan, payload)
        out.joinpath("source.json").write_text(json.dumps(source, sort_keys=True) + "\n")
    (out / "ci-request.json").write_text(json.dumps(plan, separators=(",", ":"), ensure_ascii=False) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("native_source=" + str(native).lower() + "\n")
        output.write("matrix=" + json.dumps({"include": plan["cases"]}, separators=(",", ":")) + "\n")
        output.write("sha=" + plan["sha"] + "\n")
        output.write("plan_digest=" + plan["plan_digest"] + "\n")


if __name__ == "__main__":
    main()
