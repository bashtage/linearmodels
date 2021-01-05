from setuptools import Extension, find_packages, setup
from setuptools.dist import Distribution

from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
import glob
from typing import Dict

import versioneer

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
with open("README.md", "r") as readme:
    long_description = readme.read()


additional_files = ["py.typed"]
for filename in glob.iglob("./linearmodels/datasets/**", recursive=True):
    if ".csv.bz" in filename:
        additional_files.append(filename.replace("./linearmodels/", ""))

for filename in glob.iglob("./linearmodels/tests/**", recursive=True):
    if ".txt" in filename or ".csv" in filename or ".dta" in filename:
        additional_files.append(filename.replace("./linearmodels/", ""))

for filename in glob.iglob("./examples/**", recursive=True):
    if ".png" in filename:
        additional_files.append(filename)


class BinaryDistribution(Distribution):
    def is_pure(self) -> bool:
        return False


def run_setup(binary: bool = True) -> None:
    extensions = []
    if binary:
        from Cython.Build import cythonize
        import numpy

        macros = [("NPY_NO_DEPRECATED_API", "1")]
        # macros.append(('CYTHON_TRACE', '1'))
        directives: Dict[str, bool] = {}  # {'linetrace': True, 'binding':True}
        extension = Extension(
            "linearmodels.panel._utility",
            ["linearmodels/panel/_utility.pyx"],
            define_macros=macros,
            include_dirs=[numpy.get_include()],
        )
        extensions.append(extension)
        extensions = cythonize(extensions, compiler_directives=directives, force=True)

    setup(
        cmdclass=versioneer.get_cmdclass(),
        name="linearmodels",
        license="NCSA",
        description="Linear Panel, Instrumental Variable, Asset Pricing, and System "
        "Regression models for Python",
        version=versioneer.get_version(),
        packages=find_packages(),
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
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering",
        ],
        ext_modules=extensions,
        python_requires=">=3.6",
        distclass=BinaryDistribution,
    )


try:
    run_setup(binary=True)
except (
    CCompilerError,
    DistutilsExecError,
    DistutilsPlatformError,
    IOError,
    ValueError,
    ImportError,
):
    run_setup(binary=False)
    import warnings

    warnings.warn(FAILED_COMPILER_ERROR, UserWarning)
