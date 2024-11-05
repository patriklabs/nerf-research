import os
from setuptools import setup, find_packages


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="nerf_tools",
    version="0.0.1",
    author="Patrik Persson",
    author_email="patrik.persson@live.com",
    description=("A nerf rendering tool"),
    license="MIT",
    keywords="Nerf",
    url="",
    packages=find_packages(
        include=[
            "nerf_render",
            "nerf_render.*",
            "nerf_visualizer",
            "nerf_visualizer.*",
            "nerf",
            "nerf.*",
        ]
    ),
    long_description=read("readme.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)
