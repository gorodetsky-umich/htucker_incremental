from glob import glob
import os
import sys
from setuptools import setup, find_packages

if sys.version_info[:2] < (3, 8):
    error = (
        "Htucker requires Python 3.8 or later (%d.%d detected). \n"
    )
    sys.stderr.write(error + "\n")
    sys.exit(1)


name = "htucker"
description = "Hierarchical Tucker Tensor Decomposition"
authors = {
    "Aksoy": ("Doruk Aksoy", "doruk@umich.edu"),
    "Gorodetsky": ("Alex Gorodetsky", "goroda@umich.edu"),
}

maintainer = "Alex Gorodetsky"
maintainer_email = "goroda@umich.edu"
url = None
project_urls = None
platforms = ["Linux", "Mac OSX"]
keywords = [
    "Tensor Decomposition",
    "Hierarchical Tucker",
    "Numerical Linear Algebra",
    "Scientific Computing",
    "Machine Learning",
    "Dimensionality Reduction",
    "Mathematics"
]

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

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


try:
    with open("requirements.txt") as fid:
        install_requires = [l.strip() for l in fid.readlines() if not l.startswith("#")]
except FileNotFoundError:
    install_requires = ["numpy>=1.20.0"]

with open("README.org") as fh:
    long_description = fh.read()

# Get version number
with open(os.path.join("htucker", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.strip().split("=")[1].strip(' "\'')
            break

packages = find_packages()
package_data = {
    "htucker": ["tests/*.py"],
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

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
        long_description_content_type="text/x-org",
        platforms=platforms,
        url=url,
        classifiers=classifiers,
        # project_urls=project_urls,
        packages=packages,
        # data_files=data,
        package_data=package_data,
        install_requires=install_requires,
        # extras_require=extras_require,
        python_requires=">=3.8",
        zip_safe=False,
    )
