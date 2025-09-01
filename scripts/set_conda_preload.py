#!/usr/bin/env python3
import os, sys, glob, subprocess


def split_paths(envvar):
    val = os.environ.get(envvar, "")
    return [p for p in val.split(":") if p]


def guess_patterns(user_token: str):
    """
    Build filename patterns that will match typical lib names on Linux.
    Handles inputs like: jpeg, libjpeg, libjpeg.so, libjpeg.so.9
    """
    if os.path.sep in user_token:
        # Explicit path; use as-is (and also allow suffix wildcard)
        base = user_token
        pats = [base, base + "*"]
        return pats

    # Strip leading lib and trailing .so[.N]
    token = user_token
    if token.startswith("lib"):
        token = token[3:]
    if token.endswith(".so"):
        token = token[:-3]
    # If versioned like .so.9, keep full token to match directly too
    pats = [
        f"lib{token}.so*",
        f"{token}.so*",
        f"lib{token}*",
        f"{token}*",
    ]
    # If the user passed a versioned soname, include it as-is
    if user_token.startswith("lib") and ".so." in user_token:
        pats.insert(0, user_token)
    return pats


def search_dirs():
    dirs = []
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        dirs.append(os.path.join(cp, "lib"))
    dirs += split_paths("LD_LIBRARY_PATH")
    # Allow user-specified extras
    dirs += split_paths("EXTRA_LIB_DIRS")
    # Common system locations
    dirs += [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
        "/usr/lib",
        "/lib/x86_64-linux-gnu",
        "/lib",
    ]
    # De-dup, keep order
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def find_with_ldconfig(soname_token: str):
    try:
        out = subprocess.check_output(["ldconfig", "-p"], text=True)
    except Exception:
        return None
    # Match lines containing libX.so (not just token)
    # Prefer exact lib name if user gave one
    candidates = []
    for line in out.splitlines():
        if ".so" in line and soname_token in line:
            parts = line.split("=>")
            if len(parts) == 2:
                path = parts[1].strip()
                if os.path.isfile(path):
                    candidates.append(path)
    return candidates[0] if candidates else None


def resolve_one(user_token: str):
    pats = guess_patterns(user_token)
    for d in search_dirs():
        for pat in pats:
            matches = sorted(glob.glob(os.path.join(d, pat)))
            # Prefer the longest match (e.g., libjpeg.so.9 over libjpeg.so)
            if matches:
                matches.sort(key=len, reverse=True)
                return matches[0]
    # ldconfig fallback using a likely soname
    # choose the first pattern that looks like a soname, else lib<token>.so
    soname_like = next(
        (p for p in pats if p.startswith("lib") and ".so" in p), f"lib{user_token}.so"
    )
    hit = find_with_ldconfig(soname_like.replace("*", ""))
    return hit


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: set_preload.py <lib1> [<lib2> ...]\n"
            "Examples: set_preload.py libjpeg.so libtiff.so curl",
            file=sys.stderr,
        )
        sys.exit(1)

    found = []
    for tok in sys.argv[1:]:
        path = resolve_one(tok)
        if path:
            print(f"Found {tok} -> {path}", file=sys.stderr)
            found.append(path)
        else:
            print(f"WARNING: {tok} not found", file=sys.stderr)

    if not found:
        print("# No libraries found")
        return

    preload_new = ":".join(found)
    preload_old = os.environ.get("LD_PRELOAD", "")
    if preload_old:
        preload_new = preload_new + ":" + preload_old
    print(preload_new)  # dump the preload_new variable to the stderr


if __name__ == "__main__":
    main()
