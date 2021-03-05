import os
from setuptools import setup, find_packages
import sys
sys.path.append(os.path.curdir)

PACKAGE_NAME = "episcalp"
with open(os.path.join("episcalp", "__init__.py"), "r") as fid:
    for line in (line.strip() for line in fid):
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("'").strip('"')
            break
if version is None:
    raise RuntimeError("Could not determine version")
REQUIRED_PACKAGES = [
    "numpy>=1.18",
    "scipy>=1.1.0",
    "pandas>=1.0.3",
    "pybids>=0.10",
    "pybv>=0.4.0",
    "joblib>=0.15",
    "matplotlib>=3.2.1",
    "seaborn",
    "mne>=0.22",
    "mne-bids>=0.6",
    "pyprep"
]

setup(
    name=PACKAGE_NAME,
    version=version,
    packages=find_packages(),
)
