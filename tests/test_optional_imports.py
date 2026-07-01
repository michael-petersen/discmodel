import importlib
import types

from discmodel.optional_imports import _check_flex, _check_lintsampler

def test_check_lintsampler_missing(monkeypatch):
    # make importlib.import_module raise for lintsampler
    original_import_module = importlib.import_module

    def raise_for(name):
        if name == "lintsampler":
            raise ImportError
        return original_import_module(name)
    monkeypatch.setattr(importlib, "import_module", raise_for)
    assert _check_lintsampler() is False

def test_check_lintsampler_present(monkeypatch):
    # make importlib.import_module return a dummy module for lintsampler
    original_import_module = importlib.import_module

    def stub(name):
        if name == "lintsampler":
            return types.ModuleType("lintsampler")
        return original_import_module(name)
    monkeypatch.setattr(importlib, "import_module", stub)
    assert _check_lintsampler() is True


def test_check_flex_missing(monkeypatch):
    # make importlib.import_module raise for flex
    original_import_module = importlib.import_module

    def raise_for(name):
        if name == "flex":
            raise ImportError
        return original_import_module(name)
    monkeypatch.setattr(importlib, "import_module", raise_for)
    assert _check_flex() is False


def test_check_flex_present(monkeypatch):
    # make importlib.import_module return a dummy module for flex
    original_import_module = importlib.import_module

    def stub(name):
        if name == "flex":
            return types.ModuleType("flex")
        return original_import_module(name)
    monkeypatch.setattr(importlib, "import_module", stub)
    assert _check_flex() is True
