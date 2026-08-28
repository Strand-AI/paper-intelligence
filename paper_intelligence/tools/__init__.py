"""Paper tool exports with ML-heavy modules loaded only when requested."""

from importlib import import_module

from .search import get_paper_info, grep_search, rag_search, search

_LAZY_EXPORTS = {
    "convert_pdf": ".convert",
    "index_markdown": ".index",
    "get_header_context": ".index",
    "search_headers": ".index",
    "embed_document": ".embed",
    "query_paper": ".embed",
}

__all__ = [
    *_LAZY_EXPORTS,
    "search",
    "grep_search",
    "rag_search",
    "get_paper_info",
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
