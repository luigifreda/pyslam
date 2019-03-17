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

import sys 
import os
import numpy as np
from termcolor import colored

# colors from https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])


def getchar():
    a = input('').split(" ")[0]
    print(a)


# Class to print 
# colored text and background 
# from https://www.geeksforgeeks.org/print-colors-python-terminal/
class Colors(object):
    '''
    Colors class:reset all colors with colors.reset; two  
    sub classes fg for foreground  
    and bg for background; use as colors.subclass.colorname. 
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,  
    underline, reverse, strike through, 
    and invisible work with the main class i.e. colors.bold
    '''
    reset='\033[0m'
    bold='\033[01m'
    disable='\033[02m'
    underline='\033[04m'
    reverse='\033[07m'
    strikethrough='\033[09m'
    invisible='\033[08m'
    class fg: 
        black='\033[30m'
        red='\033[31m'
        green='\033[32m'
        orange='\033[33m'
        blue='\033[34m'
        purple='\033[35m'
        cyan='\033[36m'
        lightgrey='\033[37m'
        darkgrey='\033[90m'
        lightred='\033[91m'
        lightgreen='\033[92m'
        yellow='\033[93m'
        lightblue='\033[94m'
        pink='\033[95m'
        lightcyan='\033[96m'
    class bg: 
        black='\033[40m'
        red='\033[41m'
        green='\033[42m'
        orange='\033[43m'
        blue='\033[44m'
        purple='\033[45m'
        cyan='\033[46m'
        lightgrey='\033[47m'

class Printer(object):
    @staticmethod
    def red(*args, **kwargs):
        print(Colors.fg.red, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def green(*args, **kwargs):
        print(Colors.fg.green, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def blue(*args, **kwargs):
        print(Colors.fg.blue, *args, **kwargs)
        print(Colors.reset, end="")        

    @staticmethod
    def error(*args, **kwargs):
        print(Colors.fg.red, *args, **kwargs, file=sys.stderr)
        print(Colors.reset, end="")        


# test class with termcolor
class Printer_old(object):
    @staticmethod
    def red(input):
        print(colored(input,'red'))

    @staticmethod
    def green(input):
        print(colored(input,'green'))           


# return a random RGB color tuple 
def random_color():
    color = tuple(np.random.randint(0,255,3).tolist())    
    return color 