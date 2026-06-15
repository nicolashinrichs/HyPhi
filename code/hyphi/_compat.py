"""
Shared back-compatibility unpickler for cross-version graph/array pickles.

This module centralizes the single ``_CompatUnpickler`` used to read pickles
that were written under a different NumPy or NetworkX than the importing
environment ships (issue #79). It merges the two previously divergent copies:

- the NumPy module-prefix remap (``numpy._core`` -> ``numpy.core``), so pickles
  written under ``numpy >= 1.25`` load under older releases that only expose the
  legacy ``numpy.core`` namespace; and
- the NetworkX class remap, which stubs out the legacy
  ``networkx.classes.graph.Graph`` and ``networkx.classes.reportviews.*View``
  classes referenced inside an older NetworkX pickle stream with local shims, so
  the ``_adj`` dict can be read directly without re-implementing unpickling.

``find_class`` applies both rules in order (NumPy prefix first, then the
NetworkX class remap, then the default resolution), so a single loader is robust
to old-NumPy and old-NetworkX pickles at once.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any, ClassVar

__all__ = ["CompatUnpickler", "load_compat_pickle"]


class _CompatGraph:
    """Minimal stand-in for a legacy ``networkx`` ``Graph`` referenced in a pickle.

    The shim accepts the pickle's ``__setstate__`` so the original ``_adj``
    dictionary lands in ``__dict__``, where the adjacency loader reads it.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept and ignore any positional/keyword construction arguments."""

    def __setstate__(self, state: object) -> None:
        """Restore the pickled state by updating ``__dict__`` when it is a dict."""
        if isinstance(state, dict):
            for key, value in state.items():
                self.__dict__[str(key)] = value


class _CompatView:
    """Minimal stand-in for a legacy ``networkx`` report-view referenced in a pickle.

    Report views (``NodeView``, ``EdgeView``, ...) are not needed to recover the
    adjacency, so the shim just absorbs the pickled state.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept and ignore any positional/keyword construction arguments."""

    def __setstate__(self, state: object) -> None:
        """Stash the pickled state without interpreting it."""
        self.__dict__["_state"] = state


class CompatUnpickler(pickle.Unpickler):
    """Unpickler that tolerates cross-version NumPy and NetworkX class paths.

    ``find_class`` remaps the private ``numpy._core`` namespace back onto the
    legacy ``numpy.core`` namespace, and substitutes local shims for the legacy
    NetworkX ``Graph`` and report-view classes, before falling back to the
    default class resolution.
    """

    _VIEW_NAMES: ClassVar[set[str]] = {
        "NodeView",
        "NodeDataView",
        "EdgeView",
        "OutEdgeView",
        "InEdgeView",
        "MultiEdgeView",
        "OutMultiEdgeView",
        "InMultiEdgeView",
        "EdgeDataView",
        "OutEdgeDataView",
        "InEdgeDataView",
        "MultiEdgeDataView",
        "OutMultiEdgeDataView",
        "InMultiEdgeDataView",
    }

    def find_class(self, module: str, name: str) -> object:
        """Resolve a pickled class, applying the NumPy then NetworkX remaps first.

        Parameters
        ----------
        module
            The fully qualified module name recorded in the pickle stream.
        name
            The class name recorded in the pickle stream.

        Returns
        -------
        object
            The resolved class: a remapped NumPy class, a local NetworkX shim,
            or the class found by the default resolution.
        """
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        if module == "networkx.classes.graph" and name == "Graph":
            return _CompatGraph
        if module == "networkx.classes.reportviews" and name in self._VIEW_NAMES:
            return _CompatView
        return super().find_class(module, name)


def load_compat_pickle(path: str | Path) -> Any:
    """Load a pickle through :class:`CompatUnpickler` from its own byte buffer.

    The bytes are read fully and unpickled from an in-memory buffer via the
    unpickler instance's own ``load``, so the call does not depend on the
    module-level ``pickle.load`` (which lets callers keep a fast ``pickle.load``
    path that can be monkeypatched independently of this fallback).

    Parameters
    ----------
    path
        Path to the pickle file to load.

    Returns
    -------
    Any
        The unpickled object (dynamic, as with :func:`pickle.load`).
    """
    raw = Path(path).read_bytes()
    return CompatUnpickler(io.BytesIO(raw)).load()
