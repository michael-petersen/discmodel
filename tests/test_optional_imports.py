import sys
import importlib
import pytest

def reload_discmodel():
    sys.modules.pop("discmodel.discmodel", None)
    import discmodel as dm
    importlib.reload(dm)
    return dm


def test_has_lintsampler(monkeypatch):
    monkeypatch.setitem(sys.modules, "lintsampler", object())
    dm = reload_discmodel()
    assert dm.HAS_LINTSAMPLER is True


def test_missing_lintsampler(monkeypatch):
    sys.modules.pop("lintsampler", None)
    monkeypatch.delitem(sys.modules, "lintsampler", raising=False)
    dm = reload_discmodel()
    assert dm.HAS_LINTSAMPLER is False
