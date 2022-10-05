from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

compile_args = ["-march=native", "-std=c++20"]

ext_modules = [
    Pybind11Extension(
        "zebra",
        sorted(glob("src/*.cpp")),
        include_dirs=["include", "build/_deps/vectorclass-src"],
        extra_compile_args=compile_args
    )
]

setup(
    name="zebra",
    version="0.0.1",
    description="Optimized C++ implementations of postprocessing operators.",
    author="Tierry Hörmann",
    author_email="Tierry.Hoermann@meteoswiss.ch",
    ext_modules=ext_modules,
    )