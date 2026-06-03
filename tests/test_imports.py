"""Import smoke test — every ``oracle.*`` module must import cleanly.

This guards against phantom dependencies (e.g. the former ``gstack_tools``)
re-entering the codebase and silently breaking ``import oracle.<module>`` at
module-load time. A failure here means production code cannot even be imported,
regardless of whether higher-level tests happen to exercise that path.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import oracle


def _all_oracle_modules():
    names = []
    for mod in pkgutil.walk_packages(oracle.__path__, prefix="oracle."):
        names.append(mod.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _all_oracle_modules())
def test_oracle_module_imports(module_name: str):
    """Each oracle submodule imports without raising."""
    importlib.import_module(module_name)


def test_no_gstack_dependency():
    """The phantom ``gstack_tools`` dependency must stay gone."""
    import sys

    importlib.import_module("oracle.prediction.verifier")
    importlib.import_module("oracle.resolution.resolver")
    importlib.import_module("oracle.ingestion.pipeline")
    assert "gstack_tools" not in sys.modules
