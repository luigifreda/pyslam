from setuptools import setup
from setuptools.command.install import install
from distutils.sysconfig import get_python_lib
import glob
import shutil


__library_file__ = './lib/g2o*.so'
__version__ = '0.0.1'



class CopyLibFile(install):
    """"
    Directly copy library file to python's site-packages directory.
    """

    def run(self):
        install_dir = get_python_lib()
        lib_file = glob.glob(__library_file__)
        assert len(lib_file) == 1     

        print('copying {} -> {}'.format(lib_file[0], install_dir))
        shutil.copy(lib_file[0], install_dir)




setup(
    name='g2opy',
    version=__version__,
    description='Python binding of C++ graph optimization framework g2o.',
    url='https://github.com/uoip/g2opy',
    license='BSD',
    cmdclass=dict(
        install=CopyLibFile
    ),
    keywords='g2o, SLAM, BA, ICP, optimization, python, binding',
    long_description="""This is a Python binding for c++ library g2o 
        (https://github.com/RainerKuemmerle/g2o).

        g2o is an open-source C++ framework for optimizing graph-based nonlinear 
        error functions. g2o has been designed to be easily extensible to a wide 
        range of problems and a new problem typically can be specified in a few 
        lines of code. The current implementation provides solutions to several 
        variants of SLAM and BA."""
)
