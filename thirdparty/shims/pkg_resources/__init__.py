"""
Lightweight compatibility shim for third‑party code that does:

    from pkg_resources import packaging

We don't need the full `pkg_resources` API here; the `f3rm` CLIP wrapper
only uses `packaging.version.parse`.  Importing and re‑exporting the
`packaging` module is therefore sufficient and avoids depending on how
`setuptools` bundles `pkg_resources` on a given platform/Python version.
"""

import packaging as packaging  # re-export for consumers expecting `pkg_resources.packaging`

