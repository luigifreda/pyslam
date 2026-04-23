This folder contains lightweight **Python import shims** that make a few
third-party dependencies happier on macOS (and other environments where
their expected top-level modules are missing or packaged differently).

- `pkg_resources/`:
  - Provides just enough of `pkg_resources` for code that does:
    - `from pkg_resources import packaging`
  - Internally it simply re-exports the standard `packaging` module.
  - Used to unblock:
    - The OpenAI CLIP repo during installation.
    - Libraries (e.g. `f3rm`) that call `pkg_resources.packaging.version`.

- `clip/`:
  - Minimal implementation of the `clip` package API expected by Detic and
    similar projects:
    - `import clip; clip.load(...)`
    - `from clip.simple_tokenizer import SimpleTokenizer`
  - It forwards all calls to the CLIP implementation vendored via
    `f3rm.features.clip`, so there is a **single source of truth** for the
    actual model code and weights.

These directories are **not** imported directly from the repo root. Instead,
they are copied into the active Python environment's `site-packages` by the
`scripts/install_mac_shims.sh` helper when:

- `import pkg_resources` fails, and/or
- `import clip` fails.

On Linux or properly configured environments, you typically don't need these
shims at all; the script simply detects that the real modules are available
and does nothing.
