#!/usr/bin/env python3
import os
import re
import subprocess
from pathlib import Path

# repo root = scripts/..
repo_root = Path(__file__).resolve().parent.parent
thirdparty = repo_root / "thirdparty"
install_script = repo_root / "scripts" / "install_thirdparty.sh"


def list_first_level_dirs(base: Path) -> list[str]:
    if not base.is_dir():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def git_ls_files_stage() -> list[str]:
    try:
        return subprocess.check_output(
            ["git", "ls-files", "--stage"], text=True, cwd=repo_root
        ).splitlines()
    except subprocess.CalledProcessError:
        return []


def git_submodules_under_thirdparty(ls_stage_lines: list[str]) -> list[str]:
    """
    Use `git ls-files --stage` (mode 160000) and keep only those under thirdparty/.
    """
    names = []
    for line in ls_stage_lines:
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "160000":
            path = parts[3]  # repo-relative
            if path.startswith("thirdparty/"):
                rel = path[len("thirdparty/") :]
                first = rel.split("/", 1)[0]
                if first:
                    names.append(first)
    return sorted(set(names))


def git_tracked_non_submodule_thirdparty(
    ls_stage_lines: list[str], submodules: set[str], thirdparty: Path
) -> list[str]:
    """
    Folders under thirdparty/ that have tracked files in the repo and are NOT submodules.
    Only returns names that are actual directories (ignores plain files).
    """
    tracked_folders: set[str] = set()
    for line in ls_stage_lines:
        parts = line.split()
        if len(parts) >= 4:
            mode, path = parts[0], parts[3]
            if path.startswith("thirdparty/") and mode != "160000":
                rel = path[len("thirdparty/") :]
                first = rel.split("/", 1)[0]
                if first:
                    candidate = thirdparty / first
                    if candidate.is_dir():  # ✅ ensure it's really a folder
                        tracked_folders.add(first)
    return sorted(tracked_folders - submodules)


def parse_install_script(script_path: Path) -> list[str]:
    """
    Heuristics to find which folders the install script will create under thirdparty/:
    - Matches explicit destinations:  git clone ... thirdparty/<name>
    - Handles cd'ing into thirdparty + 'git clone URL' (infer name from URL).
    """
    if not script_path.is_file():
        return []

    txt = script_path.read_text(encoding="utf-8", errors="ignore")

    cloned: set[str] = set()

    # 1) Any explicit 'thirdparty/<name>' mentions
    for m in re.finditer(r"thirdparty/([A-Za-z0-9._-]+)", txt):
        cloned.add(m.group(1))

    # 2) Infer from 'cd thirdparty' context + 'git clone <url>'
    in_thirdparty = False
    for raw_line in txt.splitlines():
        line = raw_line.strip()

        if re.search(r"\b(cd|pushd)\s+thirdparty\b", line):
            in_thirdparty = True
        if re.search(r"\b(cd\s+\.\.|popd)\b", line):
            in_thirdparty = False

        if "git clone" not in line:
            continue

        tokens = line.split()
        try:
            idx = tokens.index("clone")
        except ValueError:
            continue

        args = tokens[idx + 1 :]

        if in_thirdparty:
            repo_arg = None
            for a in args:
                if a.startswith("-"):
                    continue
                repo_arg = a
                break
            if repo_arg:
                base = repo_arg.rstrip("/").split("/")[-1]
                base = base.removesuffix(".git")
                if base:
                    cloned.add(base)

    return sorted(cloned)


if __name__ == "__main__":
    # 1) All first-level subfolders of thirdparty
    all_subfolders = list_first_level_dirs(thirdparty)

    # 2) Git info
    ls_stage_lines = git_ls_files_stage()
    submodule_dirs = git_submodules_under_thirdparty(ls_stage_lines)
    submodule_set = set(submodule_dirs)

    # 3) Folders cloned by scripts/install_thirdparty.sh
    cloned_by_script = parse_install_script(install_script)

    # 4) Non-submodules (present as directories but not submodules)
    non_submodules = sorted(set(all_subfolders) - submodule_set)

    # 5) Non-submodules NOT cloned by the script
    non_submodules_not_cloned = sorted(set(non_submodules) - set(cloned_by_script))

    # 6) NEW: thirdparty folders that are tracked in repo (non-submodules)
    tracked_non_submodule = git_tracked_non_submodule_thirdparty(
        ls_stage_lines, submodule_set, thirdparty
    )

    # Pretty print
    def header(title: str):
        print(f"\n=== {title} ===")

    header("All subfolders")
    print("\n".join(all_subfolders))

    header("Git submodules")
    print("\n".join(submodule_dirs))

    header("Cloned by scripts/install_thirdparty.sh")
    print("\n".join(cloned_by_script))

    header("Non-submodules (existing dirs not submodules)")
    print("\n".join(non_submodules))

    header("Non-submodules NOT cloned by the script")
    print("\n".join(non_submodules_not_cloned))

    header("Tracked in repo (non-submodules)")
    print("\n".join(tracked_non_submodule))
