from setuptools import setup, Extension
import pybind11
import sys

opencv_inc = r'C:/opencv/opencv/build/include'
opencv_lib = r'C:/opencv/opencv/build/x64/vc16/lib'
opencv_dll = 'opencv_world4100'   # 按你的版本改，如 opencv_world490
eigen_inc  = r'C:/eigen/eigen-5.0.0'         # header-only，解压后的根目录

if sys.platform == 'win32':
    extra_args = ['/std:c++17', '/O2']
else:
    extra_args = ['-std=c++17', '-O3']

ext = Extension(
    'slam_core',
    sources=['slam_core.cpp'],
    include_dirs=[pybind11.get_include(), opencv_inc, eigen_inc],
    library_dirs=[opencv_lib],
    libraries=[opencv_dll],
    language='c++',
    extra_compile_args=extra_args,
)

setup(name='slam_core', ext_modules=[ext])