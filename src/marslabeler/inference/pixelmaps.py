"""Temporary disk-backed maps so large HiRISE score layers do not fill RAM."""

from collections.abc import MutableMapping
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class PixelMapStore(MutableMapping):
    def __init__(self):
        self._directory = TemporaryDirectory(prefix="mars-pixel-maps-")
        self._paths = {}
        self._counter = 0

    def __getitem__(self, key):
        # np.load closes each file before returning; thousands of blocks must
        # not leave thousands of memory-mapped file descriptors open on Windows.
        return np.load(self._paths[key], allow_pickle=False)

    def __setitem__(self, key, value):
        path = Path(self._directory.name) / f"{self._counter}.npy"
        self._counter += 1
        np.save(path, np.asarray(value), allow_pickle=False)
        old = self._paths.get(key)
        self._paths[key] = path
        if old is not None:
            old.unlink(missing_ok=True)

    def __delitem__(self, key):
        self._paths.pop(key).unlink(missing_ok=True)

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)

    def __contains__(self, key):
        return key in self._paths
