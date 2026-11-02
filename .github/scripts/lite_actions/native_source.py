"""Read-only GitHub merge-source preparation; never imports PR Python."""

from __future__ import annotations

import ast
import base64
import json
import os
import re
import time
import urllib.request

from blue_catalog import BLUE_SMOKE_SUITE

REPO = "cann-framework/vllm-ascend"
REPO_ID = 1438625957
FORK = "depeng1994/vllm-ascend"
FORK_ID = 1440390070
PAYLOAD = ".github/scripts/lite_actions/blue_hccl_payload.py"
SHA = re.compile(r"[0-9a-f]{40}\Z")


def api(path):
    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise ValueError("relative repository API path required")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    request = urllib.request.Request(
        "https://api.github.com/repos/" + REPO + path,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=25) as response:
        data = response.read(2 * 1024**2 + 1)
    if len(data) > 2 * 1024**2:
        raise ValueError("source authority response exceeds budget")
    return json.loads(data)


def current_pr(number, get=api):
    if type(number) is not int or number < 1:
        raise ValueError("invalid PR number")
    pr = get(f"/pulls/{number}")
    if (
        pr.get("number") != number
        or pr.get("state") != "open"
        or pr.get("merged")
        or pr.get("draft")
        or pr.get("user", {}).get("login") != "depeng1994"
        or pr.get("base", {}).get("repo", {}).get("id") != REPO_ID
        or pr["base"]["repo"].get("full_name") != REPO
        or pr["base"].get("ref") != "main"
        or pr.get("head", {}).get("repo", {}).get("id") != FORK_ID
        or pr["head"]["repo"].get("full_name") != FORK
    ):
        raise ValueError("unauthorized Blue PR source")
    head = pr["head"].get("sha")
    base = get("/branches/main").get("commit", {}).get("sha")
    if not isinstance(head, str) or not SHA.fullmatch(head) or not isinstance(base, str) or not SHA.fullmatch(base):
        raise ValueError("invalid Head/Base SHA")
    return pr, head, base


def assert_current(source, get=api):
    if source["mode"] == "merge_result":
        pr, head, base = current_pr(source["pr_number"], get)
        if (head, base) != (source["head_sha"], source["base_sha"]):
            raise ValueError("STALE: PR Head/Base advanced; rerun from the current PR")
        labels = {x["name"] for x in pr.get("labels", [])}
        tests = list(BLUE_SMOKE_SUITE)
        if "lite-a3-blue" not in labels or tests != source["test_ids"]:
            raise ValueError("STALE: Blue selection or authorization changed")


def build_source(plan, event, get=api):
    control = os.environ["GITHUB_WORKFLOW_SHA"]
    if not SHA.fullmatch(control):
        raise ValueError("invalid immutable controller SHA")
    source = {
        "schema": 1,
        "provider": "github",
        "repository": REPO,
        "repository_id": REPO_ID,
        "run_id": plan["run_id"],
        "attempt": plan["attempt"],
        "plan_digest": plan["plan_digest"],
        "control_sha": control,
        "head_sha": plan["sha"],
        "base_ref": "main",
        "test_ids": [c["test_id"] for c in plan["cases"]],
    }
    if "pull_request" in event:
        number = event["pull_request"]["number"]
        pr, head, base = current_pr(number, get)
        if head != plan["sha"]:
            raise ValueError("STALE: event Head is no longer current")
        merge = pr.get("merge_commit_sha")
        if pr.get("mergeable") is False:
            raise ValueError("MERGE_CONFLICT")
        if not isinstance(merge, str) or not SHA.fullmatch(merge):
            raise ValueError("MERGE_NOT_READY")
        commit = get("/git/commits/" + merge)
        if commit.get("sha") != merge or [p.get("sha") for p in commit.get("parents", [])] != [base, head]:
            raise ValueError("STALE_MERGE_PARENTS")
        source.update(
            mode="merge_result",
            pr_number=number,
            head_repo_id=FORK_ID,
            base_sha=base,
            source_sha=merge,
            tree_sha=commit["tree"]["sha"],
        )
    else:
        if plan["sha"] != control:
            raise ValueError("manual source must equal its immutable controller")
        commit = get("/git/commits/" + control)
        source.update(
            mode="commit",
            pr_number=0,
            head_repo_id=REPO_ID,
            base_sha=control,
            source_sha=control,
            tree_sha=commit["tree"]["sha"],
        )
    if not SHA.fullmatch(source["tree_sha"]):
        raise ValueError("invalid tested tree")
    if any(c["kind"] != "worker_probe" for c in plan["cases"]):
        content = get("/contents/" + PAYLOAD + "?ref=" + source["source_sha"])
        if content.get("type") != "file" or content.get("path") != PAYLOAD or content.get("size", 0) > 65536:
            raise ValueError("missing/oversized source payload interface")
        tree = ast.parse(base64.b64decode(content["content"], validate=False))
        version = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "PAYLOAD_API_VERSION" for t in n.targets)
            and isinstance(n.value, ast.Constant)
            and type(n.value.value) is int
            and n.value.value == 1
            for n in tree.body
        )
        function = any(isinstance(n, ast.FunctionDef) and n.name == "all_reduce" for n in tree.body)
        if not version or not function:
            raise ValueError("source payload API 1/all_reduce interface missing")
    assert_current(source, get)
    return source


def prepare_source(plan, event):
    for attempt in range(5):
        try:
            return build_source(plan, event)
        except ValueError as exc:
            if str(exc) not in {"MERGE_NOT_READY", "STALE_MERGE_PARENTS"} or attempt == 4:
                raise
            time.sleep(2)
    raise ValueError("merge source unavailable")
