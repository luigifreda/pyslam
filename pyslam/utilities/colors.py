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

import random
import numpy as np

from typing import NamedTuple, Tuple


# return a random RGB color tuple
def random_color():
    color = tuple(np.random.randint(0, 255, 3).tolist())
    return color


class Colors:
    # colors from https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
    myjet = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.99910873],
            [0.0, 0.37843137, 1.0],
            [0.0, 0.83333333, 1.0],
            [0.30044276, 1.0, 0.66729918],
            [0.66729918, 1.0, 0.30044276],
            [1.0, 0.90123457, 0.0],
            [1.0, 0.48002905, 0.0],
            [0.99910873, 0.07334786, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    my_jet_x_255 = myjet * 255
    num_myjet_colors = myjet.shape[0]

    @staticmethod
    def myjet_color_x_255(idx: int) -> np.ndarray:
        return Colors.my_jet_x_255[idx % Colors.num_myjet_colors]


class RGB(NamedTuple):
    """RGB color tuple with 8-bit components."""

    r: int
    g: int
    b: int


class ColorTableGenerator:
    """Efficient color table generator for mapping integers to distinct colors."""

    # Customize table size if you like (keep 64..4096 reasonable)
    TABLE_SIZE = 256  # must be a power of 2
    TABLE_MASK = TABLE_SIZE - 1  # For bitwise AND instead of modulo
    USE_HASH_DISTRIBUTION = True  # set False to use raw id % TABLE_SIZE

    # Golden ratio conjugate for low-discrepancy hue distribution
    PHI = 0.61803398875

    def __init__(self):
        """Initialize the color table."""
        self._table = self._generate_table()

    @classmethod
    def instance(cls):
        """Singleton accessor."""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    def color_from_int(self, id_val: int) -> RGB:
        """Convert integer to color (fast: O(1))."""
        key = self._splitmix64(id_val) if self.USE_HASH_DISTRIBUTION else id_val
        idx = key & self.TABLE_MASK
        return self._table[idx]

    def color_from_int_without_hash(self, id_val: int) -> RGB:
        """Convert integer to color (fast: O(1))."""
        idx = id_val & self.TABLE_MASK
        return self._table[idx]

    def to_hex(self, color: RGB) -> str:
        """Return hex string like '#RRGGBB'."""
        return f"#{color.r:02X}{color.g:02X}{color.b:02X}"

    @staticmethod
    def pack_rgb(color: RGB) -> int:
        """Pack RGB to 0xRRGGBB format."""
        return (color.r << 16) | (color.g << 8) | color.b

    def _generate_table(self) -> Tuple[RGB, ...]:
        """Generate table with well-separated hues via golden-ratio stepping."""
        table = []
        h = 0.0

        for i in range(self.TABLE_SIZE):
            h = (h + self.PHI) % 1.0  # low-discrepancy hues
            r, g, b = self._hsv_to_rgb(h, 0.70, 0.90)  # vivid, readable colors
            table.append(RGB(r, g, b))

        return tuple(table)

    @staticmethod
    def _splitmix64(x: int) -> int:
        """Mix IDs so nearby integers spread across the table."""
        # Excellent 64-bit mixer; cheap and reversible enough for our use
        x += 0x9E3779B97F4A7C15
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB
        return x ^ (x >> 31)

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to RGB (returns 8-bit components)."""
        c = v * s
        hp = h * 6.0
        x = c * (1.0 - abs(hp % 2.0 - 1.0))
        m = v - c

        r = g = b = 0.0
        i = int(hp)

        if i % 6 == 0:
            r, g, b = c, x, 0
        elif i % 6 == 1:
            r, g, b = x, c, 0
        elif i % 6 == 2:
            r, g, b = 0, c, x
        elif i % 6 == 3:
            r, g, b = 0, x, c
        elif i % 6 == 4:
            r, g, b = x, 0, c
        elif i % 6 == 5:
            r, g, b = c, 0, x

        # Convert to 8-bit integers
        to_8bit = lambda u: int(round((u + m) * 255))
        return to_8bit(r), to_8bit(g), to_8bit(b)


# Convenience functions for easy access
def get_color_table_generator() -> ColorTableGenerator:
    """Get the singleton ColorTableGenerator instance."""
    return ColorTableGenerator.instance()


def int_to_color(id_val: int) -> RGB:
    """Convert integer to RGB color."""
    return get_color_table_generator().color_from_int(id_val)


def int_to_hex_color(id_val: int) -> str:
    """Convert integer to hex color string."""
    generator = get_color_table_generator()
    color = generator.color_from_int(id_val)
    return generator.to_hex(color)


class GlColors:
    kRed = np.array([1.0, 0.0, 0.0, 1.0])
    kGreen = np.array([0.0, 1.0, 0.0, 1.0])
    kBlue = np.array([0.0, 0.0, 1.0, 1.0])
    kYellow = np.array([1.0, 1.0, 0.0, 1.0])
    kCyan = np.array([0.0, 1.0, 1.0, 1.0])
    kMagenta = np.array([1.0, 0.0, 1.0, 1.0])
    kWhite = np.array([1.0, 1.0, 1.0, 1.0])
    kBlack = np.array([0.0, 0.0, 0.0, 1.0])
    kGray = np.array([0.5, 0.5, 0.5, 1.0])
    kOrange = np.array([1.0, 0.5, 0.0, 1.0])
    kPurple = np.array([0.5, 0.0, 1.0, 1.0])
    kBrown = np.array([0.5, 0.25, 0.0, 1.0])
    kTeal = np.array([0.0, 0.5, 0.5, 1.0])
    kPink = np.array([1.0, 0.5, 1.0, 1.0])
    kLime = np.array([0.5, 1.0, 0.0, 1.0])
    kIndigo = np.array([0.25, 0.0, 0.5, 1.0])
    kGold = np.array([1.0, 0.75, 0.0, 1.0])
    kMaroon = np.array([0.5, 0.0, 0.25, 1.0])
    kLavender = np.array([0.75, 0.5, 1.0, 1.0])
    kOlive = np.array([0.5, 0.5, 0.0, 1.0])
    kCoral = np.array([1.0, 0.5, 0.5, 1.0])
    kKhaki = np.array([0.75, 0.5, 0.0, 1.0])
    kAqua = np.array([0.0, 1.0, 0.5, 1.0])
    kLimeGreen = np.array([0.5, 1.0, 0.5, 1.0])
    kCrimson = np.array([1.0, 0.5, 0.5, 1.0])
    kTurquoise = np.array([0.0, 1.0, 1.0, 1.0])
    kNavy = np.array([0.0, 0.0, 0.5, 1.0])
    kViolet = np.array([0.5, 0.0, 0.5, 1.0])
    kPinkViolet = np.array([0.5, 0.5, 0.5, 1.0])
    kCyanViolet = np.array([0.5, 0.5, 0.5, 1.0])

    _instance = None

    def __init__(self):
        # get colors from static fields by iterating over them
        self.colors = [
            getattr(GlColors, color)
            for color in dir(GlColors)
            if isinstance(getattr(GlColors, color), np.ndarray) and color.startswith("k")
        ]
        self.num_colors = len(self.colors)

    @staticmethod
    def get_color(i):
        if not GlColors._instance:
            GlColors._instance = GlColors()
        return getattr(GlColors, GlColors._instance.colors[i % GlColors._instance.num_colors])

    @staticmethod
    def get_colors():
        if not GlColors._instance:
            GlColors._instance = GlColors()
        return GlColors._instance.colors

    @staticmethod
    def get_random_color():
        if not GlColors._instance:
            GlColors._instance = GlColors()
        return GlColors._instance.colors[random.randint(0, GlColors._instance.num_colors - 1)]
