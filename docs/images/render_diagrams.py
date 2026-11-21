#!/usr/bin/env python3
"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* Render the Mermaid diagram sources (*.mmd) in docs/images to PNG files.
* 
* Each <name>.mmd is rendered to <name>.png in the same folder with mermaid-cli (mmdc).
* By default only diagrams whose PNG is missing or older than the source are rendered.
* 
* Usage (from any directory):
*     python docs/images/render_diagrams.py                      # render outdated diagrams
*     python docs/images/render_diagrams.py --all                # re-render every diagram
*     python docs/images/render_diagrams.py direct_slam          # one subfolder
*     python docs/images/render_diagrams.py direct_slam/pose_tracker.mmd --scale 2
*     python docs/images/render_diagrams.py --list               # show status, render nothing
* 
* Requirements:
*     - Node.js (>= 18.19). If `mmdc` is not on PATH, `npx` fetches the pinned
*       @mermaid-js/mermaid-cli version on first use.
*     - A Chrome/Chromium browser. An installed google-chrome/chromium is used if found;
*       otherwise mermaid-cli's own Puppeteer browser is used. Override with --chrome.
* 
* Diagrams without a front-matter `config:` block get the default look below
* (hand-drawn, as on https://mermaid.live). A front-matter block in a source overrides it.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent
MERMAID_CLI_VERSION = "11.4.2"
DEFAULT_SCALE = 3
DEFAULT_BACKGROUND = "white"
DEFAULT_MERMAID_CONFIG = {"look": "handDrawn", "theme": "default"}
CHROME_CANDIDATES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")


def find_mmdc():
    """Return the command prefix that runs mermaid-cli."""
    mmdc = shutil.which("mmdc")
    if mmdc:
        return [mmdc]
    npx = shutil.which("npx")
    if npx:
        return [npx, "--yes", "-p", f"@mermaid-js/mermaid-cli@{MERMAID_CLI_VERSION}", "mmdc"]
    sys.exit("error: neither 'mmdc' nor 'npx' found; install Node.js (>= 18.19) first")


def find_chrome(explicit):
    if explicit:
        if not Path(explicit).exists():
            sys.exit(f"error: browser not found: {explicit}")
        return explicit
    for name in CHROME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None  # Let Puppeteer use its own downloaded browser.


def collect_sources(targets):
    """Resolve CLI targets (files or folders, relative to cwd or to docs/images) to .mmd files."""
    if not targets:
        return sorted(IMAGES_DIR.rglob("*.mmd"))
    sources = []
    for target in targets:
        path = Path(target)
        if not path.exists():
            path = IMAGES_DIR / target
        if path.is_dir():
            sources.extend(sorted(path.rglob("*.mmd")))
        elif path.suffix == ".mmd" and path.exists():
            sources.append(path)
        else:
            sys.exit(f"error: not a .mmd file or folder: {target}")
    return [s.resolve() for s in sources]


def display_path(path):
    try:
        return path.relative_to(IMAGES_DIR)
    except ValueError:
        return path


def is_outdated(source):
    png = source.with_suffix(".png")
    return not png.exists() or png.stat().st_mtime < source.stat().st_mtime


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help=".mmd files or folders (default: all of docs/images)")
    parser.add_argument("--all", action="store_true", help="render even if the PNG is up to date")
    parser.add_argument("--list", action="store_true", help="list diagrams and their status, then exit")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help=f"PNG scale factor (default {DEFAULT_SCALE})")
    parser.add_argument("--background", default=DEFAULT_BACKGROUND, help=f"background color (default {DEFAULT_BACKGROUND})")
    parser.add_argument("--chrome", help="path to a Chrome/Chromium executable")
    args = parser.parse_args()

    sources = collect_sources(args.targets)
    if not sources:
        sys.exit("no .mmd files found")

    if args.list:
        for source in sources:
            status = "outdated" if is_outdated(source) else "ok"
            print(f"{status:9s} {display_path(source)}")
        return

    todo = sources if args.all else [s for s in sources if is_outdated(s)]
    if not todo:
        print("All diagrams are up to date (use --all to re-render).")
        return

    mmdc = find_mmdc()
    chrome = find_chrome(args.chrome)
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        config_file = Path(tmp) / "mermaid.json"
        config_file.write_text(json.dumps(DEFAULT_MERMAID_CONFIG))
        puppeteer_file = Path(tmp) / "puppeteer.json"
        puppeteer = {"args": ["--no-sandbox"]}
        if chrome:
            puppeteer["executablePath"] = chrome
        puppeteer_file.write_text(json.dumps(puppeteer))

        for source in todo:
            png = source.with_suffix(".png")
            # Render away from the destination: a failed CLI invocation can leave a
            # partial output with a fresh timestamp, hiding the failure on the next run.
            rendered = Path(tmp) / "rendered.png"
            rendered.unlink(missing_ok=True)
            print(f"rendering {display_path(source)} -> {png.name}", flush=True)
            cmd = mmdc + [
                "-i", str(source), "-o", str(rendered),
                "-c", str(config_file), "-p", str(puppeteer_file),
                "-s", str(args.scale), "-b", args.background, "-q",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            if result.returncode != 0:
                failures.append(source)
                print((result.stderr or result.stdout).strip(), file=sys.stderr)
            else:
                # Copy before replacing so the final rename stays on the same
                # filesystem even when /tmp and the checkout are on different disks.
                with tempfile.NamedTemporaryFile(dir=png.parent, suffix=".png", delete=False) as output:
                    pending = Path(output.name)
                try:
                    shutil.copyfile(rendered, pending)
                    pending.replace(png)
                finally:
                    pending.unlink(missing_ok=True)

    print(f"done: {len(todo) - len(failures)} rendered, {len(failures)} failed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
