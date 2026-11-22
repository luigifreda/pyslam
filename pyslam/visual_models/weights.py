"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
*
* Explicit, checksum-verified model-weight resolution with a shared offline cache.
*
* No provider is imported and no network request is made until resolve() is called.
* Construction uses allow_download=False; prefetching must opt in explicitly. Existing
* wrappers keep their download behavior until individually migrated and validated.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.request import urlopen


class WeightsUnavailable(RuntimeError):
    pass


class ChecksumError(WeightsUnavailable):
    pass


@dataclass(frozen=True)
class WeightSource:
    kind: str  # hf | torch_hub | url
    filename: str
    sha256: str
    repo_id: str | None = None
    revision: str | None = None
    url: str | None = None

    def __post_init__(self):
        path = PurePosixPath(self.filename)
        if (not self.filename or path.is_absolute() or ".." in path.parts or "\\" in self.filename
                or not path.name or ":" in self.filename):
            raise ValueError("Weight filename must be a safe relative POSIX path")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", self.sha256):
            raise ValueError("Weights require a full SHA-256 digest")
        if self.kind == "hf":
            if not self.repo_id or not re.fullmatch(r"[a-fA-F0-9]{40}", self.revision or ""):
                raise ValueError("HF weights require a repo_id and an immutable 40-character commit revision")
        elif self.kind in {"url", "torch_hub"}:
            if not self.url or not self.url.startswith("https://"):
                raise ValueError("URL/torch_hub weights require an HTTPS URL")
        else:
            raise ValueError(f"Unknown weight source kind: {self.kind}")


def default_cache():
    explicit = os.environ.get("PYSLAM_MODELS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser() / "huggingface/hub"


def offline_requested():
    return any(os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
               for name in ("PYSLAM_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def registered_weights():
    """Read explicitly reviewed artifact metadata, separately from wrapper migration."""
    return json.loads(Path(__file__).with_name("verified_weights.json").read_text())


def resolve_registered_weights(name, *, cache_dir=None, offline=None, allow_download=False):
    try:
        entry = registered_weights()[name.upper()]
    except KeyError:
        raise WeightsUnavailable(f"No checksum-pinned weight artifact registered for {name!r}") from None
    sources = [WeightSource(**source) for source in entry["sources"]]
    return WeightResolver(cache_dir, offline).resolve(sources, allow_download=allow_download)


def verify(path, digest):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    if value.hexdigest().lower() != digest.lower():
        raise ChecksumError(f"SHA-256 mismatch for {path}")
    return Path(path)


@contextmanager
def _lock(path):
    # fcntl is supported on the Linux and macOS install targets. Locks cover download,
    # validation and atomic publication across processes as well as threads.
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class WeightResolver:
    def __init__(self, cache_dir=None, offline=None):
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_cache()
        self.offline = offline_requested() if offline is None else bool(offline) or offline_requested()

    def cached_path(self, source):
        return self.cache_dir / "pyslam" / source.sha256.lower() / source.filename

    def resolve(self, sources, *, allow_download=False, legacy_paths=()):
        """Resolve HF -> torch.hub checkpoint -> URL mirrors, validating every cache hit.

        All mirrors must name identical content. A legacy path is accepted only when
        its checksum matches, and is never removed or rewritten. A corrupt cache is
        reported rather than silently loaded.
        """
        sources = tuple(sources)
        if not sources:
            raise WeightsUnavailable("No verified weight source is registered")
        if len({source.sha256.lower() for source in sources}) != 1:
            raise ValueError("Weight fallbacks must resolve the same SHA-256 content")
        errors = []
        for path in legacy_paths:
            if Path(path).is_file():
                try:
                    return verify(path, sources[0].sha256)
                except ChecksumError as exc:
                    errors.append(str(exc))
        priority = {"hf": 0, "torch_hub": 1, "url": 2}
        for source in sorted(sources, key=lambda value: priority[value.kind]):
            target = self.cached_path(source)
            if target.is_file():
                # Reject a corrupt published cache immediately; do not load another
                # artifact while concealing an integrity failure.
                return verify(target, source.sha256)
            try:
                if source.kind == "hf":
                    # This import can be missing independently of the local/URL cache.
                    from huggingface_hub import hf_hub_download
                    candidate = hf_hub_download(repo_id=source.repo_id, filename=source.filename,
                                                revision=source.revision, cache_dir=str(self.cache_dir),
                                                local_files_only=self.offline or not allow_download)
                    return verify(candidate, source.sha256)
                if source.kind == "torch_hub":
                    torch_home = Path(os.environ.get("TORCH_HOME", str(
                        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "torch")))
                    candidate = torch_home / "hub/checkpoints" / PurePosixPath(source.filename).name
                    if candidate.is_file():
                        return verify(candidate, source.sha256)
                if self.offline or not allow_download:
                    errors.append(f"{source.kind}:{source.filename}: not cached (offline)")
                    continue
                return self._download(source, target)
            except ChecksumError:
                raise
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                errors.append(f"{source.kind}:{source.filename}: {exc}")
        raise WeightsUnavailable("Weights unavailable. Prefetch explicitly or provide a verified legacy path. "
                                 + "; ".join(errors))

    def _download(self, source, target):
        with _lock(target.with_suffix(target.suffix + ".lock")):
            if target.is_file():
                return verify(target, source.sha256)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                    temporary = Path(output.name)
                    with urlopen(source.url, timeout=30) as response:
                        shutil.copyfileobj(response, output)
                verify(temporary, source.sha256)
                temporary.replace(target)
                return target
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
