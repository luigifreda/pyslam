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
* Build visual TUM-VI reference executables without changing the PLVS2 checkout.
*
* Uses an existing PLVS2 CMake build's flags and libraries. The example source is
* copied into --output. Loop closure is disabled through the runtime API because
* some PLVS2 builds crash when loopClosing: 0 leaves the loop queue unconsumed.
* Keep loopClosing: 1 in the reference YAML. Viewer/lines/dense mapping must be off.
"""

import argparse
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plvs2-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, folder in (('mono_tum_vi', 'Monocular'), ('stereo_tum_vi', 'Stereo')):
        build = args.plvs2_root.resolve() / 'build'
        target = build / 'CMakeFiles' / (name + '.dir')
        source = (args.plvs2_root / 'Examples' / folder / (name + '.cc')).read_text()
        anchor = 'float imageScale = SLAM.GetImageScale();'
        if source.count(anchor) != 1:
            raise ValueError('Unsupported PLVS2 example layout')
        source = source.replace(anchor, 'SLAM.DisableLoopClosing();\n    cv::setNumThreads(1);\n    ' + anchor)
        local_source, obj, executable = (args.output / (name + suffix) for suffix in ('.cc', '.o', ''))
        local_source.write_text(source)
        flags = {}
        for line in (target / 'flags.make').read_text().splitlines():
            if ' = ' in line:
                key, value = line.split(' = ', 1)
                flags[key] = shlex.split(value)
        command = ['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'],
                   *flags['CXX_FLAGS'], '-c', str(local_source), '-o', str(obj)]
        subprocess.run(command, cwd=build, check=True)
        command = shlex.split((target / 'link.txt').read_text())
        for i, item in enumerate(command):
            if item.endswith('.cc.o'):
                command[i] = str(obj)
            if i > 0 and command[i-1] == '-o':
                command[i] = str(executable)
        subprocess.run(command, cwd=build, check=True)
        print(executable, flush=True)


if __name__ == '__main__':
    main()
