# Zebra

Zebra provides high-performance single-core implementations of finch operators.
Both a C++ interface (cebra), as well as a Python interface (zebra) are provided.

# Building and installing

## Required toolchain versions

- CMake 3.14+
- C++ compiler supporting at least c++20 standard with concepts (e.g. GCC 10+)
- Python 3.10+

## CMake

The targets can be built using CMake.

```
mkdir build && cd build
cmake ..
cmake --build .
```

Three targets are defined:
- zebra: A python extension module which can be importet in a normal python script
- cebra: A C++ library
- zebra_test: An executable for running tests

## Setuptools

For installation of zebra via `pip`, a `setup.py` script is provided.
You can install the library by navigating to the project root and running
```
pip install .
```

# Practice tips

The finch conda environment contains the necessary toolchain.
It is recommended to use it.

## VSCode

VSCode with CMake Tools must be properly configured in order to work with the toolchain provided by a conda environment.
First set the path of CMake to the one cmake executable provided by conda in your workspace settings.
For example:
```
"cmake.cmakePath": "/users/tierriminator/miniconda3/envs/finch/bin/cmake"
```
You can find the appropriate cmake path by running `which cmake` in a terminal with an activated conda environment.
You must probably reload the window for the changes to be effective.

Now you must create an appropriate CMake Kit, because CMake Tools cannot find the conda provided compiler by itself, as explained in [this](https://github.com/microsoft/vscode-cmake-tools/issues/1445) issue.

Run `CMake: Edit User-Local CMake Kits` from `Ctrl-Shift-P` and add a new entry according to your conda installation directory.
For example:
```
{
    "name": "finch conda env",
    "environmentSetupScript": "/users/tierriminator/miniconda3/etc/profile.d/conda.sh\" ; conda activate finch\""
}
```
Now you can select the `finch conda env` kit which loads the conda-provided compilers as defaults.

If still an error pops up, it can often be solved by deleting the `build` directory.
