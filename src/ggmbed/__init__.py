from .encoder import Embedder

try:
    from importlib.metadata import version
    __version__ = version("ggmbed")
except Exception:
    __version__ = "0.2.0"

__all__ = ["Embedder"]
