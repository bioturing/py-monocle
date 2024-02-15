import os
from setuptools import setup


__DIRNAME__ = os.path.dirname(os.path.abspath(__file__))


BASE_PACKAGE = "py-monocle"
BASE_IMPORT = "py_monocle"


def _get_version():
  version_path = os.path.join(__DIRNAME__, "py_monocle3", "_version.py")
  variables = {}
  with open(version_path, encoding="utf-8") as f:
    # pylint: disable=exec-used
    # type: ignore # pylint: disable=undefined-variable
    exec(f.read(), None, variables)
  return variables["__version__"]


def _install_requires():
  with open(os.path.join(__DIRNAME__, "requirements.txt"), "r") as rf:
    return list(map(str.strip, rf.readlines()))


setup(
    name=BASE_PACKAGE,
    version=_get_version(),
    author="BioTuring",
    author_email="support@bioturing.com",
    url="https://alpha.bioturing.com",
    description="BioTuring Py-Monocle3",
    long_description="",
    package_dir={BASE_IMPORT: "py_monocle3"},
    packages=[BASE_IMPORT],
    zip_safe=False,
    python_requires=">=3.8, <3.12",
    install_requires=_install_requires(),
)
