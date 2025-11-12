
def _check_lintsampler():
    try:
        import importlib
        importlib.import_module("lintsampler")
        return True
    except ImportError:
        return False
    
def _check_flex():
    try:
        import importlib
        importlib.import_module("lintsampler")
        return True
    except ImportError:
        return False
    