"""Runtime compatibility shims for launching the original TPB code on Windows.

This file is injected with PYTHONPATH for smoke runs only; it does not modify the
original TPB checkout.
"""

import os

if not hasattr(os, "POSIX_FADV_DONTNEED"):
    os.POSIX_FADV_DONTNEED = 0

try:
    import yaml
except Exception:
    yaml = None

if yaml is not None:
    _orig_load = yaml.load

    def _compat_load(stream, Loader=None, *args, **kwargs):
        if Loader is None:
            Loader = yaml.SafeLoader
        return _orig_load(stream, Loader=Loader, *args, **kwargs)

    yaml.load = _compat_load

try:
    from torch_geometric.data import Dataset
except Exception:
    Dataset = None

if Dataset is not None:
    def _compat_len(self):
        return self.__len__()

    def _compat_get(self, idx):
        return self.__getitem__(idx)

    if not hasattr(Dataset, "len"):
        Dataset.len = _compat_len
    else:
        Dataset.len = _compat_len
    if not hasattr(Dataset, "get"):
        Dataset.get = _compat_get
    else:
        Dataset.get = _compat_get
    Dataset.__abstractmethods__ = frozenset()
