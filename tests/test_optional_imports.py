import importlib
import types

from discmodel.optional_imports import _check_flex, _check_lintsampler

def test_check_lintsampler_missing(monkeypatch):
    # make importlib.import_module raise for lintsampler
    def raise_for(name):
        if name == "lintsampler":
            raise ImportError
        return importlib.import_module(name)
    monkeypatch.setattr(importlib, "import_module", raise_for)
    assert _check_lintsampler() is False

def test_check_lintsampler_present(monkeypatch):
    # make importlib.import_module return a dummy module for lintsampler
    def stub(name):
        if name == "lintsampler":
            return types.ModuleType("lintsampler")
        return importlib.import_module(name)
    monkeypatch.setattr(importlib, "import_module", stub)
    assert _check_lintsampler() is True


def test_check_flex_missing(monkeypatch):
    # make importlib.import_module raise for flex
    def raise_for(name):
        if name == "flex":
            raise ImportError
        return importlib.import_module(name)
    monkeypatch.setattr(importlib, "import_module", raise_for)
    assert _check_flex() is False


def test_check_flex_present(monkeypatch):
    # make importlib.import_module return a dummy module for flex
    def stub(name):
        if name == "flex":
            return types.ModuleType("flex")
        return importlib.import_module(name)
    monkeypatch.setattr(importlib, "import_module", stub)
    assert _check_flex() is True
