from glob import glob
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension
import pathlib

class CMakeExtension(Extension):

    def __init__(self, name, path: str):
        super().__init__(name, sources=[])
        self.path = path

class cmake_build_ext(build_ext):

    def run(self) -> None:
        for ext in self.extensions:
            self.build_cmake(ext)
        return super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.parent.mkdir(parents=True, exist_ok=True)

        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config,
            "-DPYTHON_EXECUTABLE=" + sys.executable
        ]

        build_args = [
            '--config', config,
            "--target", ext.name,
            "--", "-j4"
        ]

        subprocess.check_call(["cmake", str(cwd)] + cmake_args, cwd=build_temp)
        if not self.dry_run:
            subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

compile_args = ["-march=native", "-std=c++20"]

ext_modules = [
    # CMakeExtension(
    #     "zebra",
    #     "."
    # )
    Pybind11Extension(
        "zebra",
        sorted(glob("src/*.cpp")),
        include_dirs=["include", "build/_deps/vectorclass-src"],
        compile_args=compile_args
    )
]

setup(
    name="zebra",
    version="0.0.1",
    description="Optimized C++ implementations of postprocessing operators.",
    author="Tierry Hörmann",
    author_email="Tierry.Hoermann@meteoswiss.ch",
    ext_modules=ext_modules,
    # cmdclass={
    #     "build_ext": cmake_build_ext
    #     }
)