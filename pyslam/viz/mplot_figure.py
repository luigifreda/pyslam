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

import time
import numpy as np
import math

import matplotlib.pyplot as plt

# refer to https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure


class MPlotFigure:
    def __init__(self, img, title=None, scale=1, dpi=100):
        self.dpi = dpi
        self.width = round(img.shape[0] * scale / dpi)
        self.height = round(img.shape[1] * scale / dpi)
        # self.fig = plt.figure(dpi=self.dpi, figsize=(self.height,self.width), tight_layout=True, frameon=False)
        self.fig = plt.figure(dpi=self.dpi, tight_layout=True, frameon=True)
        self.fig.set_facecolor("white")
        if title is not None:
            self.fig.suptitle(title, bbox={"facecolor": "white", "edgecolor": "none"})
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    # actually show the figures
    @staticmethod
    def show():
        # Use non-blocking show
        plt.show(block=False)

        # Wait for user input to close
        input("Press Enter to close all windows and exit...")
        plt.close("all")

    # make it full screen
    def full_screen(self):
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()
