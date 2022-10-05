from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

compile_args = ["-march=native", "-std=c++20"]

ext_modules = [
    Pybind11Extension(
        "postprocc",
        sorted(glob("postproc_ops/src/*.cpp")),
        include_dirs=["postproc_ops/include", "postproc_ops/build/_deps/vectorclass-src"],
        extra_compile_args=compile_args
    )
]

setup(
    name="postprocc",
    version="0.0.1",
    description="Optimized C++ implementations of postprocessing operators.",
    author="Tierry Hörmann",
    author_email="Tierry.Hoermann@meteoswiss.ch",
    ext_modules=ext_modules,
    )