import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools.cleanup_non_impel_artifacts as cleanup_module
from tools.cleanup_non_impel_artifacts import (
    NON_IMPEL_MODEL_DIRS,
    find_non_impel_artifact_dirs,
    is_safe_cleanup_target,
    remove_dirs,
)


class CleanupNonImpelArtifactsTest(unittest.TestCase):
    def test_find_non_impel_artifact_dirs_only_selects_generated_model_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for city in ["Delivery_SH", "Delivery_HZ"]:
                for name in ["dcrnn", "grin", "gwnet", "ignnk", "satcn", "stgcn", "impel"]:
                    (root / "logs" / city / name).mkdir(parents=True)
            (root / "data" / "Delivery_SH").mkdir(parents=True)
            (root / "src" / "models").mkdir(parents=True)

            found = find_non_impel_artifact_dirs(root)
            rel = sorted(path.relative_to(root).as_posix() for path in found)

            self.assertIn("logs/Delivery_SH/dcrnn", rel)
            self.assertIn("logs/Delivery_HZ/stgcn", rel)
            self.assertNotIn("logs/Delivery_SH/impel", rel)
            self.assertFalse(any(item.startswith("data/") for item in rel))
            self.assertFalse(any(item.startswith("src/") for item in rel))
            self.assertEqual(
                sorted(NON_IMPEL_MODEL_DIRS),
                ["dcrnn", "grin", "gwnet", "ignnk", "satcn", "stgcn"],
            )

    def test_remove_dirs_dry_run_does_not_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "logs" / "Delivery_SH" / "dcrnn"
            target.mkdir(parents=True)

            removed = remove_dirs([target], dry_run=True)

            self.assertEqual([target], removed)
            self.assertTrue(target.exists())

    def test_remove_dirs_execute_deletes_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "logs" / "Delivery_SH" / "dcrnn"
            target.mkdir(parents=True)

            removed = remove_dirs([target], dry_run=False)

            self.assertEqual([target], removed)
            self.assertFalse(target.exists())

    def test_remove_dirs_refuses_target_resolving_outside_logs_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            logs_root = root / "logs"
            outside_target = Path(tmpdir) / "external" / "dcrnn"
            logs_root.mkdir(parents=True)
            outside_target.mkdir(parents=True)

            removed = remove_dirs([outside_target], dry_run=False, logs_root=logs_root)

            self.assertEqual([], removed)
            self.assertTrue(outside_target.exists())

    def test_safe_cleanup_target_requires_lexical_logs_path_and_resolved_repo_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            safe_target = root / "logs" / "Delivery_SH" / "dcrnn"
            outside_target = Path(tmpdir) / "external" / "Delivery_SH" / "dcrnn"
            repo_data_target = root / "data" / "Delivery_SH" / "dcrnn"
            data_traversal_target = root / "logs" / ".." / "data" / "Delivery_SH" / "dcrnn"
            traversal_target = root / "logs" / ".." / ".." / "external" / "Delivery_SH" / "dcrnn"
            safe_target.mkdir(parents=True)
            outside_target.mkdir(parents=True)
            repo_data_target.mkdir(parents=True)

            self.assertTrue(is_safe_cleanup_target(root, safe_target))
            self.assertFalse(is_safe_cleanup_target(root, outside_target))
            self.assertFalse(is_safe_cleanup_target(root, repo_data_target))
            self.assertFalse(is_safe_cleanup_target(root, data_traversal_target))
            self.assertFalse(is_safe_cleanup_target(root, traversal_target))

    def test_safe_cleanup_target_rejects_resolved_path_inside_repo_but_outside_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            logs_root = root / "logs"
            lexical_target = logs_root / "Delivery_SH" / "dcrnn"
            resolved_outside_logs = root / "data" / "Delivery_SH" / "dcrnn"
            logs_root.mkdir(parents=True)
            resolved_outside_logs.mkdir(parents=True)

            original_resolve = cleanup_module.Path.resolve

            def fake_resolve(path, *args, **kwargs):
                if path == lexical_target:
                    return resolved_outside_logs
                return original_resolve(path, *args, **kwargs)

            with patch.object(cleanup_module.Path, "resolve", fake_resolve):
                self.assertFalse(is_safe_cleanup_target(root, lexical_target))

    def test_symlinked_logs_root_outside_repo_returns_no_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            root = tmp_root / "repo"
            external_logs = tmp_root / "external_logs"
            external_target = external_logs / "Delivery_SH" / "dcrnn"
            root.mkdir()
            external_target.mkdir(parents=True)

            link = root / "logs"
            try:
                link.symlink_to(external_logs, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlink creation is unavailable: {exc}")

            found = find_non_impel_artifact_dirs(root)
            removed = remove_dirs(found, dry_run=False, root=root)

            self.assertEqual([], found)
            self.assertEqual([], removed)
            self.assertTrue(external_target.exists())

    def test_link_like_logs_root_returns_no_targets_before_scanning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logs_root = root / "logs"
            (logs_root / "Delivery_SH" / "dcrnn").mkdir(parents=True)

            def fake_is_link_like(path):
                return Path(path) == logs_root

            with patch.object(cleanup_module, "_is_link_like", side_effect=fake_is_link_like):
                found = find_non_impel_artifact_dirs(root)

            self.assertEqual([], found)

    def test_symlinked_city_dir_outside_logs_is_not_selected_or_deleted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            root = tmp_root / "repo"
            external_city = tmp_root / "external" / "Delivery_SH"
            external_target = external_city / "dcrnn"
            logs_root = root / "logs"
            logs_root.mkdir(parents=True)
            external_target.mkdir(parents=True)

            link = logs_root / "Delivery_SH"
            try:
                link.symlink_to(external_city, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlink creation is unavailable: {exc}")

            found = find_non_impel_artifact_dirs(root)
            removed = remove_dirs(found, dry_run=False)

            self.assertEqual([], found)
            self.assertEqual([], removed)
            self.assertTrue(external_target.exists())


if __name__ == "__main__":
    unittest.main()
