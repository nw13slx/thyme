from setuptools import setup, find_packages
from pathlib import Path

package_names = ["thyme"]
_name = "_".join(package_names)
name = "-".join(package_names)

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / _name / "_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name=name,
    version="0.0.1",
    author="Lixin Sun",
    python_requires=">=3.8",
    packages=find_packages(include=[name, _name, _name + ".*"]),
    install_requires=[
        "numpy",
        "ase",
    ],
    zip_safe=True,
)
