from setuptools import Extension, find_namespace_packages, setup
from setuptools.dist import Distribution
from setuptools.errors import CCompilerError, ExecError, PlatformError

import glob
import os

try:
    from Cython.Build import cythonize

    CYTHON_INSTALLED = True
except ImportError:
    CYTHON_INSTALLED = False


FAILED_COMPILER_ERROR = """
******************************************************************************
*                               WARNING                                      *
******************************************************************************

Unable to build binary modules for linearmodels.  These are not required
to run any code in the package, and only provide speed-ups for a small subset
of models.

******************************************************************************
*                               WARNING                                      *
******************************************************************************
"""
with open("README.md") as readme:
    long_description = readme.read()


additional_files = ["py.typed"]
for filename in glob.iglob("linearmodels/datasets/**", recursive=True):
    if ".csv.bz" in filename:
        additional_files.append(filename.replace("./linearmodels/", ""))

for filename in glob.iglob("linearmodels/tests/**", recursive=True):
    if ".txt" in filename or ".csv" in filename or ".dta" in filename:
        additional_files.append(filename.replace("./linearmodels/", ""))

for filename in glob.iglob("./examples/**", recursive=True):
    if ".png" in filename:
        additional_files.append(filename)

additional_files.extend(glob.glob("linearmodels/**/*.pyx", recursive=True))
additional_files.extend(glob.glob("linearmodels/**/*.pyi", recursive=True))
additional_files.extend(glob.glob("linearmodels/**/*.pyd", recursive=True))
additional_files.extend(glob.glob("linearmodels/**/*.so", recursive=True))


class BinaryDistribution(Distribution):
    def is_pure(self) -> bool:
        return False


def run_setup(binary: bool = True) -> None:
    extensions = []
    if binary:
        import numpy

        macros = [("NPY_NO_DEPRECATED_API", "1")]
        # macros.append(('CYTHON_TRACE', '1'))
        directives: dict[str, bool] = {}  # {'linetrace': True, 'binding':True}
        extension = Extension(
            "linearmodels.panel._utility",
            ["linearmodels/panel/_utility.pyx"],
            define_macros=macros,
            include_dirs=[numpy.get_include()],
        )
        extensions.append(extension)
        extensions = cythonize(extensions, compiler_directives=directives, force=True)
    else:
        import logging

        logging.warning("Building without binary support")

    setup(
        name="linearmodels",
        license="NCSA",
        description="Linear Panel, Instrumental Variable, Asset Pricing, and System "
        "Regression models for Python",
        packages=["linearmodels"]
        + [f"linearmodels.{v}" for v in find_namespace_packages("linearmodels")],
        package_dir={"linearmodels": "./linearmodels"},
        author="Kevin Sheppard",
        author_email="kevin.k.sheppard@gmail.com",
        url="http://github.com/bashtage/linearmodels",
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=open("requirements.txt").read().split("\n"),
        include_package_data=True,
        package_data={"linearmodels": additional_files},
        keywords=[
            "linear models",
            "regression",
            "instrumental variables",
            "IV",
            "panel",
            "fixed effects",
            "clustered",
            "heteroskedasticity",
            "endogeneity",
            "instruments",
            "statistics",
            "statistical inference",
            "econometrics",
        ],
        zip_safe=False,
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: End Users/Desktop",
            "Intended Audience :: Financial and Insurance Industry",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: University of Illinois/NCSA Open Source License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering",
        ],
        ext_modules=extensions,
        python_requires=">=3.9",
        distclass=BinaryDistribution,
    )


try:
    build_binary = CYTHON_INSTALLED
    build_binary &= os.environ.get("LM_NO_BINARY", None) not in ("1", "True", "true")
    run_setup(binary=build_binary)
except (
    CCompilerError,
    ExecError,
    PlatformError,
    OSError,
    ValueError,
    ImportError,
):
    run_setup(binary=False)
    import warnings

    warnings.warn(FAILED_COMPILER_ERROR, UserWarning)
