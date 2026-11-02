"""Blue producer/merge-source regressions without importing source Python."""

import base64
import copy
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import native_reconcile  # noqa: E402 - isolated script-directory import for controller tests
import native_result  # noqa: E402 - isolated script-directory import for controller tests
import native_source  # noqa: E402 - isolated script-directory import for controller tests
import prepare_request  # noqa: E402 - isolated script-directory import for controller tests
import wait_result  # noqa: E402 - isolated script-directory import for controller tests


class BlueSourceTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, GITHUB_WORKFLOW_SHA="c" * 40, GITHUB_RUN_ID="123")
        self.env.start()
        self.addCleanup(self.env.stop)
        self.base = "b" * 40
        self.pr = dict(
            number=7,
            state="open",
            draft=False,
            merged=False,
            mergeable=True,
            merge_commit_sha="d" * 40,
            user={"login": "depeng1994"},
            labels=[{"name": "lite-a3-blue"}],
            head=dict(
                sha="a" * 40,
                repo=dict(id=1440390070, full_name="depeng1994/vllm-ascend"),
            ),
            base=dict(
                ref="main",
                repo=dict(id=1438625957, full_name="cann-framework/vllm-ascend"),
            ),
        )
        self.parents = [{"sha": self.base}, {"sha": "a" * 40}]
        self.payload = (
            b"PAYLOAD_API_VERSION = 1\ndef all_reduce(tensor):\n    raise RuntimeError('not executed in Prepare')\n"
        )
        self.event = {"pull_request": self.pr}
        self.plan = prepare_request.create_plan("pull_request_target", self.event, "", 123, 1)

    def get(self, path):
        if path == "/pulls/7":
            return self.pr
        if path == "/branches/main":
            return {"commit": {"sha": self.base}}
        if path.startswith("/git/commits/"):
            return {
                "sha": path.rsplit("/", 1)[-1],
                "parents": self.parents,
                "tree": {"sha": "e" * 40},
            }
        if path.startswith("/contents/"):
            return dict(
                type="file",
                path=native_source.PAYLOAD,
                size=len(self.payload),
                content=base64.b64encode(self.payload).decode(),
            )
        raise AssertionError(path)

    def build(self):
        return native_source.build_source(self.plan, self.event, self.get)

    def test_blue_plan_keeps_schema3_and_fixed_resources(self):
        self.assertEqual(self.plan["schema"], 3)
        self.assertEqual(self.plan["sha"], "a" * 40)
        self.assertEqual(self.plan["cases"][0]["resources"], dict(nodes=0, devices_per_node=0))
        self.assertIsNone(self.plan["cases"][0]["docker"])
        self.assertEqual(self.plan["cases"][1]["docker"], "vllm_hccl_smoke")
        self.assertEqual(self.plan["cases"][1]["resources"], dict(nodes=1, devices_per_node=8))

    def test_legacy_blue16_label_is_redundant_and_does_not_duplicate_case(self):
        self.pr["labels"].append({"name": "lite-a3-16p"})
        plan = prepare_request.create_plan("pull_request_target", self.event, "", 123, 1)
        self.assertEqual(plan, self.plan)
        self.assertEqual(plan["cases"][-1]["test_id"], "dispatch_smoke_a3_16p")
        self.assertEqual(plan["cases"][-1]["resources"], dict(nodes=2, devices_per_node=8))

    def test_default_blue_requires_all_three_cases_without_16p_label(self):
        self.assertEqual(
            [x["test_id"] for x in self.plan["cases"]],
            ["vllm_worker_os_blue", "dispatch_smoke_a3_8p", "dispatch_smoke_a3_16p"],
        )
        self.assertEqual(self.plan["cases"][2]["resources"], {"nodes": 2, "devices_per_node": 8})
        self.assertEqual(self.plan["cases"][2]["docker"], "vllm_hccl_smoke")
        self.assertEqual(self.plan["cases"][2]["worker"], "blue")

    def test_previous_os_8p_only_result_is_stale_under_full_suite(self):
        source = self.build()
        source["test_ids"] = source["test_ids"][:2]
        with self.assertRaisesRegex(ValueError, "selection"):
            native_source.assert_current(source, self.get)

    def test_manual_three_case_selection_remains_available(self):
        ids = ["vllm_worker_os_blue", "dispatch_smoke_a3_8p", "dispatch_smoke_a3_16p"]
        with patch.dict(os.environ, GITHUB_REF="refs/heads/main", GITHUB_SHA="c" * 40):
            plan = prepare_request.create_plan(
                "workflow_dispatch", {}, json.dumps([{"test_id": x} for x in ids]), 123, 1
            )
        self.assertEqual([x["test_id"] for x in plan["cases"]], ids)

    def test_real_source_identity_is_merge_not_head(self):
        source = self.build()
        self.assertEqual(source["head_sha"], "a" * 40)
        self.assertEqual(source["base_sha"], "b" * 40)
        self.assertEqual(source["source_sha"], "d" * 40)
        self.assertEqual(source["tree_sha"], "e" * 40)
        self.assertEqual(source["control_sha"], "c" * 40)
        self.assertEqual(source["plan_digest"], self.plan["plan_digest"])

    def test_python_payload_is_only_ast_inspected_never_executed(self):
        self.payload = b"raise RuntimeError('control must not execute this')\nPAYLOAD_API_VERSION = 1\ndef all_reduce(tensor): pass\n"
        self.assertEqual(self.build()["source_sha"], "d" * 40)

    def test_merge_conflict_is_a_prepare_failure(self):
        self.pr["mergeable"] = False
        with self.assertRaisesRegex(ValueError, "MERGE_CONFLICT"):
            self.build()

    def test_stale_platform_merge_parent_is_rejected(self):
        self.parents[0]["sha"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "STALE_MERGE_PARENTS"):
            self.build()

    def test_missing_payload_interface_is_prepare_failure(self):
        for payload in (
            b"PAYLOAD_API_VERSION = 2\ndef all_reduce(t): pass\n",
            b"PAYLOAD_API_VERSION = True\ndef all_reduce(t): pass\n",
            b"PAYLOAD_API_VERSION = 1\n",
        ):
            self.payload = payload
            with self.assertRaisesRegex(ValueError, "interface"):
                self.build()

    def test_authority_change_after_build_invalidates(self):
        source = self.build()
        self.base = "f" * 40
        with self.assertRaisesRegex(ValueError, "STALE"):
            native_source.assert_current(source, self.get)

    def test_spec_upgrade_and_label_removal_invalidate_old_green(self):
        source = self.build()
        for labels in ([], [{"name": "lite-a3-16p"}]):
            self.pr["labels"] = labels
            with self.assertRaisesRegex(ValueError, "selection"):
                native_source.assert_current(source, self.get)

    def test_external_fork_or_retarget_cannot_dispatch(self):
        self.pr["head"]["repo"]["id"] = 1
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            self.build()

    def test_manual_blue_and_a5_cannot_mix(self):
        selections = json.dumps([{"test_id": "dispatch_smoke_a3_8p"}, {"test_id": "dispatch_smoke_a5_8p"}])
        with patch.dict(os.environ, GITHUB_REF="refs/heads/main", GITHUB_SHA="c" * 40):
            with self.assertRaisesRegex(ValueError, "cannot share"):
                prepare_request.create_plan("workflow_dispatch", {}, selections, 123, 1)

    def test_waiter_requires_identical_sidecar(self):
        source = self.build()
        case = self.plan["cases"][1]
        result = dict(
            schema=3,
            repo=prepare_request.REPO,
            workflow=prepare_request.WORKFLOW,
            run_id=123,
            attempt=1,
            sha="a" * 40,
            plan_digest=self.plan["plan_digest"],
            **{
                k: case[k]
                for k in (
                    "case_id",
                    "test_id",
                    "worker",
                    "target_pool",
                    "docker",
                    "resources",
                )
            },
            status="PASS",
            exit_code=0,
            cleanup_verified=True,
            last_20_lines=[],
            native_source=source,
        )

        def find(value):
            return wait_result.find_result(
                [
                    dict(
                        user={"login": "depeng1994"},
                        body="Lite-CI-V3 123:1:" + case["case_id"] + "\n" + json.dumps(value),
                    )
                ],
                run_id=123,
                attempt=1,
                sha="a" * 40,
                digest=self.plan["plan_digest"],
                item=case,
                workflow=prepare_request.WORKFLOW,
                author="depeng1994",
                native_source=source,
            )

        self.assertEqual(find(result), result)
        wrong = copy.deepcopy(result)
        wrong["native_source"]["source_sha"] = "f" * 40
        self.assertIsNone(find(wrong))
        wrong = {k: v for k, v in result.items() if k != "native_source"}
        self.assertIsNone(find(wrong))

    def test_cancelled_publication_cannot_write_status(self):
        source = self.build()
        with (
            patch.object(native_result, "assert_current"),
            patch.object(
                native_result,
                "api",
                return_value=dict(status="completed", run_attempt=1),
            ),
            patch.object(native_result, "post") as post,
        ):
            with self.assertRaisesRegex(ValueError, "cannot publish"):
                native_result.publish(source, "success")
            post.assert_not_called()

    def test_base_reconcile_invalidates_before_dispatch_and_deduplicates(self):
        self.base = "c" * 40
        status = []
        calls = []

        def get(path):
            if path.startswith("/pulls?"):
                return [self.pr]
            if path.startswith("/commits/"):
                return {"statuses": status}
            return self.get(path)

        def send(path, data):
            calls.append((path, data))
            if path.startswith("/statuses/"):
                status.append(data)
            return {"workflow_run_id": 789}

        result = native_reconcile.reconcile(get, send)
        self.assertTrue(calls[0][0].startswith("/statuses/"))
        self.assertEqual(calls[0][1]["state"], "pending")
        self.assertEqual(
            calls[0][1]["description"],
            native_result.description(
                self.base, ["vllm_worker_os_blue", "dispatch_smoke_a3_8p", "dispatch_smoke_a3_16p"]
            ),
        )
        self.assertTrue(calls[1][0].endswith("/dispatches"))
        self.assertEqual(calls[1][1]["inputs"]["expected_base_sha"], "c" * 40)
        self.assertEqual(result[0]["run_id"], 789)
        native_reconcile.reconcile(get, send)
        self.assertEqual(len(calls), 2)

    def test_workflow_pins_all_control_checkouts_and_uploads_request_last(self):
        root = SCRIPTS.parents[2]
        workflow = (root / ".github/workflows/lite_actions_dispatcher_smoke.yml").read_text()
        self.assertEqual(workflow.count("ref: ${{ github.workflow_sha }}"), 3)
        self.assertLess(
            workflow.index("name: lite-ci-source-"),
            workflow.index("name: lite-ci-request-"),
        )
        self.assertNotIn("ref: ${{ github.event.pull_request.head", workflow)


if __name__ == "__main__":
    unittest.main()
