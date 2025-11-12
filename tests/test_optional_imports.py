# tests/test_optional_imports.py
import importlib
import pytest
import discmodel.discmodel as dm

def test_check_lintsampler_missing(monkeypatch):
    # make importlib.import_module raise for lintsampler
    def raise_for(name):
        if name == "lintsampler":
            raise ImportError
        return importlib.import_module(name)
    monkeypatch.setattr(importlib, "import_module", raise_for)
    assert dm._check_lintsampler() is False

def test_check_lintsampler_present(monkeypatch):
    # make importlib.import_module return a dummy module for lintsampler
    def stub(name):
        if name == "lintsampler":
            import types
            return types.ModuleType("lintsampler")
        return importlib.import_module(name)
    monkeypatch.setattr(importlib, "import_module", stub)
    assert dm._check_lintsampler() is True
