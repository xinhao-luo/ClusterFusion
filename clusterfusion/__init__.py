import importlib

def _get_extension_name():
    return "_clusterfusion"

try:
    _ext = importlib.import_module(f".{_get_extension_name()}", __package__)
except ImportError as e:
    raise ImportError(
        "Failed to import clusterfusion native extension. "
        "Please ensure the package is built and installed correctly."
    ) from e

for attr in dir(_ext):
    if not attr.startswith("_"):
        globals()[attr] = getattr(_ext, attr)

del importlib
del _get_extension_name
del _ext
