#!/usr/bin/env bash
# Author: Luigi Freda
# This file is part of https://github.com/luigifreda/pyslam
#
# Prefetch the code snapshots and weights of visual models integrated through pinned
# sources (DeDoDe, RDD, RoMa, tiny-RoMa, SALAD, BoQ, ...) and verify their registered
# checksums. Existing integrations keep their own installers (install_thirdparty.sh, ...).
#
# Usage:
#   ./scripts/install_visual_model.sh --list
#   ./scripts/install_visual_model.sh ROMA SALAD
#   ./scripts/install_visual_model.sh --group vpr
#   ./scripts/install_visual_model.sh --dry-run

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$SCRIPT_DIR_/.."
cd "$ROOT_DIR" || exit 1
python -m pyslam.visual_models install "$@"
