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
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# use mplotlib figure to draw in 3D trajectories

kPlotSleep = 0.0001


class Mplot3d:
    def __init__(self, title=""):
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection="3d")
        if title is not "":
            self.ax.set_title(title)
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        self.ax.set_zlabel("Z axis")

        self.axis_computed = False
        self.xlim = [float("inf"), float("-inf")]
        self.ylim = [float("inf"), float("-inf")]
        self.zlim = [float("inf"), float("-inf")]

        self.handle_map = {}
        self.setAxis()

    def setAxis(self):
        self.ax.axis("equal")
        if self.axis_computed:
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
            self.ax.set_zlim(self.zlim)
        self.ax.legend()

    def draw(self, traj, name, color="r", marker="."):
        np_traj = np.asarray(traj)
        if name in self.handle_map:
            handle = self.handle_map[name]
            self.ax.collections.remove(handle)
        self.updateMinMax(np_traj)
        handle = self.ax.scatter3D(
            np_traj[:, 0], np_traj[:, 1], np_traj[:, 2], c=color, marker=marker
        )
        handle.set_label(name)
        self.handle_map[name] = handle

    def updateMinMax(self, np_traj):
        xmax, ymax, zmax = np.amax(np_traj, axis=0)
        xmin, ymin, zmin = np.amin(np_traj, axis=0)
        cx = 0.5 * (xmax + xmin)
        cy = 0.5 * (ymax + ymin)
        cz = 0.5 * (zmax + zmin)
        if False:
            # update maxs
            if xmax > self.xlim[1]:
                self.xlim[1] = xmax
            if ymax > self.ylim[1]:
                self.ylim[1] = ymax
            if zmax > self.zlim[1]:
                self.zlim[1] = zmax
            # update mins
            if xmin < self.xlim[0]:
                self.xlim[0] = xmin
            if ymin < self.ylim[0]:
                self.ylim[0] = ymin
            if zmin < self.zlim[0]:
                self.zlim[0] = zmin
        # make axis actually squared
        if True:
            # smin = min(self.xlim[0],self.ylim[0],self.zlim[0])
            # smax = max(self.xlim[1],self.ylim[1],self.zlim[1])
            smin = min(xmin, ymin, zmin)
            smax = max(xmax, ymax, zmax)
            delta = 0.5 * (smax - smin)
            self.xlim = [cx - delta, cx + delta]
            self.ylim = [cy - delta, cy + delta]
            self.zlim = [cz - delta, cz + delta]
        self.axis_computed = True

    def refresh(self):
        self.setAxis()
        plt.pause(kPlotSleep)
