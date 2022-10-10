from re import M
from skbuild import setup

def exclude_static_libraries(cmake_manifest: list[str]):
    return list(filter(lambda name: not (name.endswith(".a")), cmake_manifest))

setup(
    name="zebra",
    version="0.0.1",
    description="Optimized C++ implementations of postprocessing operators.",
    author="Tierry Hörmann",
    author_email="Tierry.Hoermann@meteoswiss.ch",
    # cmake_install_dir="zebra",
    cmake_process_manifest_hook=exclude_static_libraries,
    cmake_minimum_required_version="3.14"
    )
