from distutils.core import setup
import finch

setup(
    name="finch",
    version=finch.version,
    description="Experiment framework for parallelized data processing",
    author="Tierry Hörmann",
    author_email="Tierry.Hoermann@meteoswiss.ch",
    url="https://github.com/MeteoSwiss-APN/finch",
    packages=["finch", "finch.brn", "zebra"]
)