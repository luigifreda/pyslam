"""
Local shim for the `clip` package expected by Detic and other third‑party
code.  Instead of installing OpenAI's CLIP from GitHub (which currently
fails to build in this environment), we re‑use the CLIP implementation
already installed via `f3rm`.

This exposes a minimal API compatible with:

    import clip
    from clip.simple_tokenizer import SimpleTokenizer

and:

    model, preprocess = clip.load(...)
"""

from f3rm.features.clip import clip as _f3rm_clip
from f3rm.features.clip.simple_tokenizer import SimpleTokenizer

# Re‑export the tokenizer and the main entry points so that
# `import clip; clip.load(...)` works as expected.
load = _f3rm_clip.load
tokenize = _f3rm_clip.tokenize

__all__ = ["load", "tokenize", "SimpleTokenizer"]

