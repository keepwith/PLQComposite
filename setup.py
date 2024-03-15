import os
from pathlib import Path
import zipfile
import requests
from setuptools import setup
from setuptools.command.build_ext import build_ext

__version__ = "0.0.1"

# The directory that contains setup.py
SETUP_DIRECTORY = Path(__file__).resolve().parent

setup(
    name="PLQ Composite Decomposition",
    version=__version__,
    author=["Tingxian Gao", "Ben Dai"],
    author_email="txgao@link.cuhk.edu.hk",
    url="https://github.com/keepwith/PLQComposite",
    description="Piecewise Linear Quadratic Function Decomposition to Regularized Composite ReLU-ReHU Loss",
    packages=["plqcom"],
    install_requires=["requests", "numpy", "rehline", "sympy"],
    # extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
