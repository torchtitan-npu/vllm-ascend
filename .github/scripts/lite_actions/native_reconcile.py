"""Compensate Base-only updates using trusted main code and bounded API reads.

A pending status is a durable dispatch-intent claim. Never blindly retry POST
when its response was lost: an operator can explicitly re-run a full PR intent.
This deliberately favors no duplicate hardware execution over guessing success.
"""

import hashlib
import json
import os

from native_result import CONTEXT, description, post
from native_source import REPO, api, current_pr

WORKFLOW = "lite_actions_dispatcher_smoke.yml"


def reconcile(get=api, send=post):
    control = os.environ["GITHUB_WORKFLOW_SHA"]
    if get("/branches/main")["commit"]["sha"] != control:
        raise ValueError("reconciler control is no longer main")
    pulls = get("/pulls?state=open&per_page=100")
    if not isinstance(pulls, list) or len(pulls) >= 100:
        raise ValueError("open PR visibility budget incomplete")
    results = []
    for item in pulls:
        labels = {x["name"] for x in item.get("labels", [])}
        if (
            item.get("draft")
            or "lite-a3-blue" not in labels
            or item.get("user", {}).get("login") != "depeng1994"
            or item.get("head", {}).get("repo", {}).get("full_name")
            != "depeng1994/vllm-ascend"
        ):
            continue
        number = item["number"]
        pr, head, base = current_pr(number, get)
        labels = {x["name"] for x in pr.get("labels", [])}
        if "lite-a3-blue" not in labels:
            continue
        tests = ["vllm_worker_os_blue", "dispatch_smoke_a3_8p"]
        if "lite-a3-16p" in labels:
            tests.append("dispatch_smoke_a3_16p")
        desc = description(base, tests)
        statuses = get("/commits/" + head + "/status?per_page=100").get("statuses", [])
        prior = next((x for x in statuses if x.get("context") == CONTEXT), None)
        if prior and prior.get("description") == desc:
            results.append(
                {"pr": number, "state": "EXISTING_VERSION_OR_DISPATCH_UNKNOWN"}
            )
            continue
        nonce = hashlib.sha256(
            json.dumps(
                [number, head, base, control, tests], separators=(",", ":")
            ).encode()
        ).hexdigest()[:32]
        # Invalidate old green before creating the new version's execution.
        send(
            "/statuses/" + head,
            {
                "state": "pending",
                "context": CONTEXT,
                "description": desc,
                "target_url": f"https://github.com/{REPO}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        )
        response = send(
            f"/actions/workflows/{WORKFLOW}/dispatches",
            {
                "ref": "main",
                "return_run_details": True,
                "inputs": {
                    "source_pr": str(number),
                    "expected_head_sha": head,
                    "expected_base_sha": base,
                    "bridge_nonce": nonce,
                },
            },
        )
        results.append(
            {
                "pr": number,
                "state": "DISPATCHED",
                "nonce": nonce,
                "run_id": response.get("workflow_run_id")
                if isinstance(response, dict)
                else None,
            }
        )
    return results


if __name__ == "__main__":
    print(json.dumps(reconcile(), sort_keys=True))
