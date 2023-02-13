from glob import glob
import os
import sys
from setuptools import setup

if sys.version_info[:2] < (3, 8):
    error = (
        "Htucker requires Python 3.8 or later (%d.%d detected). \n"
    )
    sys.stderr.write(error + "\n")
    sys.exit(1)


name = "htucker"
description = "Hierarchical Tucker"
authors = {
    "Gorodetsky": ("Alex Gorodetsky", "goroda@umich.edu"),
}

maintainer = "Alex Gorodetsky"
maintainer_email = "goroda@umich.edu"
url = None
project_urls = None
# url = "https://networkx.org/"
# project_urls = {
#     "Bug Tracker": "https://github.com/networkx/networkx/issues",
#     "Documentation": "https://networkx.org/documentation/stable/",
#     "Source Code": "https://github.com/networkx/networkx",
# }
platforms = ["Linux", "Mac OSX"]
keywords = [
    "Networks",
    "Graph Theory",
    "Mathematics",
    "network",
    "graph",
    "discrete mathematics",
    "math",
]

classifiers = []
# classifiers = [
#     "Development Status :: 5 - Production/Stable",
#     "Intended Audience :: Developers",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: BSD License",
#     "Operating System :: OS Independent",
#     "Programming Language :: Python :: 3",
#     "Programming Language :: Python :: 3.8",
#     "Programming Language :: Python :: 3.9",
#     "Programming Language :: Python :: 3.10",
#     "Programming Language :: Python :: 3.11",
#     "Programming Language :: Python :: 3 :: Only",
#     "Topic :: Software Development :: Libraries :: Python Modules",
#     "Topic :: Scientific/Engineering :: Bio-Informatics",
#     "Topic :: Scientific/Engineering :: Information Analysis",
#     "Topic :: Scientific/Engineering :: Mathematics",
#     "Topic :: Scientific/Engineering :: Physics",
# ]

with open("htucker/__init__.py") as fid:
    for line in fid:
        if line.startswith("__version__"):
            version = line.strip().split()[-1][1:-1]
            break

packages = [
    "htucker",
    # "networkx.algorithms",
    # "networkx.algorithms.assortativity", 
]

# add the tests subpackage(s)
package_data = {
    "htucker": ["tests/*.py"],
    # "networkx.algorithms": ["tests/*.py"], 
}


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if not l.startswith("#")]

    return requires


install_requires = []
# extras_require = {
#     dep: parse_requirements_file("requirements/" + dep + ".txt")
#     for dep in ["default", "developer", "doc", "extra", "test"]
# } 

with open("README.org") as fh:
    long_description = fh.read()

if __name__ == "__main__":

    setup(
        name=name,
        version=version,
        maintainer=maintainer,
        maintainer_email=maintainer_email,
        author=authors["Gorodetsky"][0],
        author_email=authors["Gorodetsky"][1],
        description=description,
        keywords=keywords,
        long_description=long_description,
        platforms=platforms,
        url=url,
        # project_urls=project_urls,
        classifiers=classifiers,
        packages=packages,
        # data_files=data,
        package_data=package_data,
        install_requires=install_requires,
        # extras_require=extras_require,
        python_requires=">=3.8",
        zip_safe=False,
    )
