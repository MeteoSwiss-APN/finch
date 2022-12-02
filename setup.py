from skbuild import setup

def exclude_static_libraries(cmake_manifest: list[str]):
    return list(filter(lambda name: not (name.endswith(".a")), cmake_manifest))

setup(
    packages = [
        "finch",
        "finch.brn"
    ],
    cmake_source_dir = "zebra",
    cmake_process_manifest_hook = exclude_static_libraries,
    cmake_minimum_required_version="3.14",
    cmake_args=["-DINSTALL_GTEST=OFF"]
)