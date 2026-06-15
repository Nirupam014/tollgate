"""Optional tree-sitter backend for real graph recovery in non-Python languages.

The core analyzer is stdlib-only. This module is the *only* place that touches
tree-sitter, and it is imported lazily: if the `multilang` extra
(`pip install "tollgate[multilang]"`) isn't installed, `available()` is False and
every caller honest-fails to the advisory textual lint — never a fabricated graph.

Tree-sitter parsing is deterministic, which keeps Tollgate's reproducibility
guarantee intact.
"""
from __future__ import annotations

from typing import List, Optional

# Map our language ids / file kinds to tree-sitter-language-pack names.
_LANG = {"go": "go", "golang": "go", "java": "java", "ruby": "ruby", "rb": "ruby"}


def available() -> bool:
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_language_pack  # noqa: F401
        return True
    except Exception:
        return False


def get_parser(language: str):
    from tree_sitter_language_pack import get_parser as _gp
    return _gp(_LANG.get(language, language))


def parse(source: str, language: str):
    """Parse source to a tree-sitter Tree, or None if the backend is unavailable."""
    if not available():
        return None
    return get_parser(language).parse(bytes(source, "utf-8"))


def node_text(node) -> str:
    """Source text of a node (version-tolerant)."""
    t = getattr(node, "text", None)
    if t is not None:
        return t.decode("utf-8", "ignore") if isinstance(t, (bytes, bytearray)) else str(t)
    return ""


def walk(node):
    """Yield every node in the tree (pre-order)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.named_children))


def sexp(node, indent: int = 0, max_depth: int = 60) -> str:
    """Indented s-expression of the *named* nodes, with leaf text — a
    grammar-agnostic dump used by scripts/ts_probe.py to discover node types."""
    pad = "  " * indent
    label = node.type
    if not node.named_children:
        txt = node_text(node).replace("\n", "\\n")
        if txt and txt != label:
            label += f"  {txt[:48]!r}"
        return pad + label
    out: List[str] = [pad + label]
    if indent >= max_depth:
        return "\n".join(out)
    for ch in node.named_children:
        out.append(sexp(ch, indent + 1, max_depth))
    return "\n".join(out)
