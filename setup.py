from skbuild import setup

def exclude_static_libraries(cmake_manifest: list[str]):
    return list(filter(lambda name: not (name.endswith(".a")), cmake_manifest))

def get_version() -> str:
    with open("finch/data/VERSION") as f:
        return f.readline().strip()

setup(
    version = get_version(),
    packages = [
        "finch",
        "finch.brn"
    ],
    include_package_data=True,
    scripts = [
        "scripts/finch"
    ],
    cmake_source_dir = "zebra",
    cmake_process_manifest_hook = exclude_static_libraries,
    cmake_minimum_required_version="3.14",
    cmake_args=["-DINSTALL_GTEST=OFF"]
)