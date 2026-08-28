"""Utility exports, loaded lazily so lightweight tools avoid ML imports."""

from importlib import import_module

_EXPORTS = {
    "RAGClient": ".chromadb_client",
    "create_documents_from_markdown": ".chromadb_client",
    "MarkdownParser": ".markdown_parser",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
