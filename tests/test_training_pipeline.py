import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class TrainingPipelineTest(unittest.TestCase):
    def test_build_pipeline_plan_excludes_target_and_names_outputs(self):
        from experiments.training.run_multisource_dpo_pipeline import build_pipeline_plan

        plan = build_pipeline_plan(
            cities="Delivery_SH,Delivery_HZ,Delivery_CQ",
            target_city="Delivery_HZ",
            run_root="results/training",
            run_tag="unit",
        )

        self.assertEqual(plan["target_city"], "Delivery_HZ")
        self.assertEqual(plan["source_cities"], ["Delivery_SH", "Delivery_CQ"])
        self.assertEqual(plan["run_dir"], str(Path("results/training") / "unit"))
        self.assertEqual(plan["stages"][0]["output"], str(Path("results/training") / "unit" / "backbone" / "best_backbone.pt"))
        self.assertEqual(
            [stage["name"] for stage in plan["stages"]],
            [
                "backbone",
                "experts",
                "candidate_cache",
                "supervised_router",
                "dpo_router",
                "zero_shot_eval",
            ],
        )
        for stage in plan["stages"][:-1]:
            self.assertNotIn("Delivery_HZ", stage["uses_cities"])
        self.assertEqual(plan["stages"][-1]["uses_cities"], ["Delivery_HZ"])

    def test_write_dry_run_plan_creates_readable_summary(self):
        from experiments.training.run_multisource_dpo_pipeline import build_pipeline_plan, write_dry_run_plan

        with tempfile.TemporaryDirectory() as tmpdir:
            plan = build_pipeline_plan(
                cities="Delivery_SH,Delivery_HZ",
                target_city="Delivery_HZ",
                run_root=tmpdir,
                run_tag="dry",
            )
            path = write_dry_run_plan(plan)

            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("target_city: Delivery_HZ", text)
            self.assertIn("source_cities: Delivery_SH", text)
            self.assertIn("dpo_router", text)

    def test_run_pipeline_executes_stages_in_order_and_writes_status(self):
        from experiments.training.run_multisource_dpo_pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            calls = []

            def fake_stage_runner(stage, args, plan):
                calls.append(stage["name"])
                Path(stage["output"]).parent.mkdir(parents=True, exist_ok=True)
                Path(stage["output"]).write_text(stage["name"], encoding="utf-8")
                return {"stage": stage["name"]}

            result = run_pipeline(
                SimpleNamespace(
                    cities="Delivery_SH,Delivery_HZ,Delivery_CQ",
                    target_city="Delivery_HZ",
                    run_root=tmpdir,
                    run_tag="run",
                    dry_run=False,
                    resume=False,
                    skip_backbone=False,
                    skip_experts=False,
                    skip_dpo=False,
                    expert_config="",
                ),
                stage_runner=fake_stage_runner,
            )

            self.assertEqual(calls, ["backbone", "experts", "candidate_cache", "supervised_router", "dpo_router", "zero_shot_eval"])
            self.assertEqual(result["status"]["zero_shot_eval"]["status"], "completed")
            self.assertTrue((Path(tmpdir) / "run" / "pipeline_status.json").exists())

    def test_run_pipeline_skip_dpo_evaluates_supervised_router(self):
        from experiments.training.run_multisource_dpo_pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            seen_eval_router = []

            def fake_stage_runner(stage, args, plan):
                if stage["name"] == "zero_shot_eval":
                    seen_eval_router.append(stage["inputs"]["router_ckpt"])
                Path(stage["output"]).parent.mkdir(parents=True, exist_ok=True)
                Path(stage["output"]).write_text(stage["name"], encoding="utf-8")

            result = run_pipeline(
                SimpleNamespace(
                    cities="Delivery_SH,Delivery_HZ,Delivery_CQ",
                    target_city="Delivery_HZ",
                    run_root=tmpdir,
                    run_tag="skip",
                    dry_run=False,
                    resume=False,
                    skip_backbone=False,
                    skip_experts=False,
                    skip_dpo=True,
                    expert_config="",
                ),
                stage_runner=fake_stage_runner,
            )

            self.assertNotIn("dpo_router", result["status"])
            self.assertTrue(seen_eval_router[0].endswith(str(Path("supervised_router") / "best_router.pt")))

    def test_run_pipeline_resume_skips_existing_stage_output(self):
        from experiments.training.run_multisource_dpo_pipeline import build_pipeline_plan, run_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            plan = build_pipeline_plan(
                cities="Delivery_SH,Delivery_HZ,Delivery_CQ",
                target_city="Delivery_HZ",
                run_root=tmpdir,
                run_tag="resume",
            )
            first_output = Path(plan["stages"][0]["output"])
            first_output.parent.mkdir(parents=True, exist_ok=True)
            first_output.write_text("existing", encoding="utf-8")
            calls = []

            def fake_stage_runner(stage, args, plan):
                calls.append(stage["name"])
                Path(stage["output"]).parent.mkdir(parents=True, exist_ok=True)
                Path(stage["output"]).write_text(stage["name"], encoding="utf-8")

            run_pipeline(
                SimpleNamespace(
                    cities="Delivery_SH,Delivery_HZ,Delivery_CQ",
                    target_city="Delivery_HZ",
                    run_root=tmpdir,
                    run_tag="resume",
                    dry_run=False,
                    resume=True,
                    skip_backbone=False,
                    skip_experts=False,
                    skip_dpo=False,
                    expert_config="",
                ),
                stage_runner=fake_stage_runner,
            )

            self.assertNotIn("backbone", calls)
            self.assertIn("experts", calls)

    def test_run_pipeline_missing_output_fails_and_records_status(self):
        from experiments.training.run_multisource_dpo_pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "did not produce"):
                run_pipeline(
                    SimpleNamespace(
                        cities="Delivery_SH,Delivery_HZ,Delivery_CQ",
                        target_city="Delivery_HZ",
                        run_root=tmpdir,
                        run_tag="fail",
                        dry_run=False,
                        resume=False,
                        skip_backbone=False,
                        skip_experts=False,
                        skip_dpo=False,
                        expert_config="",
                    ),
                    stage_runner=lambda stage, args, plan: None,
                )

            status_path = Path(tmpdir) / "fail" / "pipeline_status.json"
            self.assertTrue(status_path.exists())
            self.assertIn("failed", status_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
