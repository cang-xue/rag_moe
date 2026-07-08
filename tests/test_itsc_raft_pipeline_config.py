import tempfile
import unittest
from pathlib import Path

import yaml

from tools.write_itsc_raft_pipeline_config import write_pipeline_config


class ITSCRAFTPipelineConfigTest(unittest.TestCase):
    def test_write_pipeline_config_combines_itsc_and_calibrated_raft(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            itsc = root / "itsc.yaml"
            raft = root / "raft.yaml"
            output = root / "experts_itsc_raft.yaml"
            itsc.write_text(
                "experts:\n"
                "  itsc:\n"
                "    class: ITSCExpert\n"
                "    mode: full_model\n"
                "    checkpoint_path: original.pt\n"
                "    bank_path: ./data/rag_bank/itsc/Delivery_SH_source_bank_5000_v5.pkl\n"
                "  tpb:\n"
                "    class: TPBExpert\n"
            )
            raft.write_text(
                "experts:\n"
                "  raft:\n"
                "    class: RAFTExpert\n"
                "    mode: final_prediction\n"
                "    bank_path: ./data/rag_bank/raft/Delivery_SH_temporal_shape_bank.pt\n"
                "    prior_alpha: 0.25\n"
                "  tpb:\n"
                "    class: TPBExpert\n"
            )

            write_pipeline_config(itsc, raft, output)

            text = output.read_text()
            config = yaml.safe_load(text)
            self.assertEqual(sorted(config["experts"]), ["itsc", "raft"])
            self.assertIn("    mode: residual\n", text)
            self.assertEqual(config["experts"]["itsc"]["mode"], "residual")
            self.assertEqual(config["experts"]["raft"]["mode"], "residual")
            self.assertEqual(config["experts"]["raft"]["prior_alpha"], 0.25)

    def test_write_pipeline_config_rejects_non_mapping_yaml_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            itsc = root / "itsc.yaml"
            raft = root / "raft.yaml"
            output = root / "experts_itsc_raft.yaml"
            itsc.write_text("- not\n- a mapping\n")
            raft.write_text("experts:\n  raft:\n    class: RAFTExpert\n")

            with self.assertRaisesRegex(ValueError, "YAML root must be a mapping"):
                write_pipeline_config(itsc, raft, output)

    def test_write_pipeline_config_rejects_missing_experts_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            itsc = root / "itsc.yaml"
            raft = root / "raft.yaml"
            output = root / "experts_itsc_raft.yaml"
            itsc.write_text("metadata:\n  city: SH\n")
            raft.write_text("experts:\n  raft:\n    class: RAFTExpert\n")

            with self.assertRaisesRegex(ValueError, "must define an experts mapping"):
                write_pipeline_config(itsc, raft, output)

    def test_write_pipeline_config_rejects_null_selected_expert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            itsc = root / "itsc.yaml"
            raft = root / "raft.yaml"
            output = root / "experts_itsc_raft.yaml"
            itsc.write_text("experts:\n  itsc:\n")
            raft.write_text("experts:\n  raft:\n    class: RAFTExpert\n")

            with self.assertRaisesRegex(
                ValueError, "must define experts\\.itsc as a mapping"
            ):
                write_pipeline_config(itsc, raft, output)

    def test_write_pipeline_config_rejects_missing_selected_expert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            itsc = root / "itsc.yaml"
            raft = root / "raft.yaml"
            output = root / "experts_itsc_raft.yaml"
            itsc.write_text("experts:\n  raft:\n    class: RAFTExpert\n")
            raft.write_text("experts:\n  raft:\n    class: RAFTExpert\n")

            with self.assertRaisesRegex(
                ValueError, "must define experts\\.itsc as a mapping"
            ):
                write_pipeline_config(itsc, raft, output)

    def test_write_pipeline_config_rejects_non_mapping_selected_expert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            itsc = root / "itsc.yaml"
            raft = root / "raft.yaml"
            output = root / "experts_itsc_raft.yaml"
            itsc.write_text("experts:\n  itsc: invalid\n")
            raft.write_text("experts:\n  raft:\n    class: RAFTExpert\n")

            with self.assertRaisesRegex(
                ValueError, "must define experts\\.itsc as a mapping"
            ):
                write_pipeline_config(itsc, raft, output)


if __name__ == "__main__":
    unittest.main()
