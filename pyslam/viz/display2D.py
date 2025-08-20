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

# import pygame
print("importing pygame")
import pygame

print("initialising pygame")
pygame.init()

from pygame.locals import DOUBLEBUF


class Display2D(object):
    def __init__(self, W, H, is_BGR=True):
        pygame.init()
        pygame.display.set_caption("Camera")
        self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.is_BGR = is_BGR

    def quit(self):
        pygame.display.quit()
        pygame.quit()

    def draw(self, img):
        # junk
        for event in pygame.event.get():
            pass

        if self.is_BGR:
            # draw BGR
            pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1)[:, :, [2, 1, 0]])
        else:
            # draw RGB, not BGR
            pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1)[:, :, [0, 1, 2]])

        self.screen.blit(self.surface, (0, 0))

        # blit
        pygame.display.flip()

    def get_key(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                return event.key
