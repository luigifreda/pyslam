#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
Commonly used functions
"""

from datetime import datetime


class ClassProperty(property):
    """For dynamically obtaining system time"""

    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class Notify(object):
    """Colorful printing prefix.
    A quick example:
    print(Notify.INFO, YOUR TEXT, Notify.ENDC)
    """

    def __init__(self):
        pass

    @ClassProperty
    def HEADER(cls):
        return str(datetime.now()) + ': \033[95m'

    @ClassProperty
    def INFO(cls):
        return str(datetime.now()) + ': \033[92mI'

    @ClassProperty
    def OKBLUE(cls):
        return str(datetime.now()) + ': \033[94m'

    @ClassProperty
    def WARNING(cls):
        return str(datetime.now()) + ': \033[93mW'

    @ClassProperty
    def FAIL(cls):
        return str(datetime.now()) + ': \033[91mF'

    @ClassProperty
    def BOLD(cls):
        return str(datetime.now()) + ': \033[1mB'

    @ClassProperty
    def UNDERLINE(cls):
        return str(datetime.now()) + ': \033[4mU'
    ENDC = '\033[0m'
