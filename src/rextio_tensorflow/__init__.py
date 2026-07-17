"""rextio-tensorflow: public Alpha plugin for a tiny TensorFlow slice.

Implements Rextio plugin API 1.3 for a float32 CPU inference surface lowered
via an owned thin safe wrapper over the active ``tensorflow==2.21.0`` wheel's
TFE C API (dlopen/dlsym; no abandoned high-level TensorFlow Rust crate).

The package root re-exports the plugin facade eagerly; that facade defers core
analyzer/config/plugin-host imports so generated runtimes can still import
``rextio_tensorflow.types`` under a minimal ``rextio`` package.
"""

from rextio_tensorflow.__about__ import __version__
from rextio_tensorflow.plugin import RextioTensorflowPlugin, plugin

__all__ = ["RextioTensorflowPlugin", "__version__", "plugin"]
