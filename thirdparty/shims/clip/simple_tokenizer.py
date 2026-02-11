"""
Compatibility wrapper so that third‑party code can do:

    from clip.simple_tokenizer import SimpleTokenizer

We forward this to the implementation provided by `f3rm`.
"""

from f3rm.features.clip.simple_tokenizer import SimpleTokenizer  # noqa: F401

__all__ = ["SimpleTokenizer"]

