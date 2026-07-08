import argparse
import os
import shutil
from pathlib import Path


NON_IMPEL_MODEL_DIRS = {"dcrnn", "grin", "gwnet", "ignnk", "satcn", "stgcn"}


def _absolute_lexical(path):
    return Path(os.path.abspath(path))


def _path_is_within(path, root):
    try:
        Path(path).relative_to(Path(root))
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _resolved_path_is_within(path, root):
    try:
        return _path_is_within(Path(path).resolve(), Path(root).resolve())
    except (OSError, RuntimeError):
        return False


def _is_link_like(path):
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (is_junction is not None and is_junction())


def _resolves_to_self(path):
    try:
        return Path(path).resolve() == _absolute_lexical(path)
    except (OSError, RuntimeError):
        return False


def is_safe_cleanup_target(root, target):
    root_lexical = _absolute_lexical(root)
    logs_lexical = root_lexical / "logs"
    target_lexical = _absolute_lexical(target)

    if not _path_is_within(target_lexical, logs_lexical):
        return False
    logs_root = _safe_logs_root(root_lexical)
    if logs_root is None:
        return False
    if not _resolved_path_is_within(target_lexical, logs_root):
        return False
    return True


def _safe_logs_root(root):
    root_lexical = _absolute_lexical(root)
    logs_root = root_lexical / "logs"
    if not logs_root.exists():
        return None
    if _is_link_like(logs_root):
        return None
    if not logs_root.is_dir():
        return None
    if not _resolves_to_self(logs_root):
        return None
    if not _resolved_path_is_within(logs_root, root_lexical):
        return None
    return logs_root


def find_non_impel_artifact_dirs(root):
    root = _absolute_lexical(root)
    logs_root = _safe_logs_root(root)
    if logs_root is None:
        return []

    targets = []
    for city_dir in logs_root.iterdir():
        if _is_link_like(city_dir):
            continue
        if not city_dir.is_dir():
            continue
        if not is_safe_cleanup_target(root, city_dir):
            continue
        for model_name in sorted(NON_IMPEL_MODEL_DIRS):
            candidate = city_dir / model_name
            if candidate.is_dir() and is_safe_cleanup_target(root, candidate):
                targets.append(candidate)
    return sorted(targets)


def remove_dirs(paths, dry_run=True, root=None, logs_root=None):
    removed = []
    if root is None and logs_root is not None:
        root = Path(logs_root).parent
    for path in paths:
        path = Path(path)
        if root is not None and not is_safe_cleanup_target(root, path):
            continue
        removed.append(path)
        if not dry_run:
            shutil.rmtree(path)
    return removed


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--execute", action="store_true", help="Delete selected directories.")
    return parser


def main():
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    targets = find_non_impel_artifact_dirs(root)
    removed = remove_dirs(targets, dry_run=not args.execute, root=root)
    action = "would remove" if not args.execute else "removed"
    for path in removed:
        print(f"{action}: {path}")
    print(f"{action} {len(removed)} directories")


if __name__ == "__main__":
    main()
