import glob
import os

from setuptools import setup, find_packages

import versioneer

# Copy over notebooks from examples to docs for build
notebooks = glob.glob('examples/*.ipynb')
for nb in notebooks:
    fname = os.path.split(nb)[-1]
    folder, nbname = fname.split('_')
    outfile = os.path.join('doc', 'source', folder, nbname)
    with open(outfile, 'w') as nbout:
        with open(nb, 'r') as nbin:
            nbout.write(nbin.read())

setup(
    cmdclass=versioneer.get_cmdclass(),
    name='linearmodels',
    version=versioneer.get_version(),
    packages=find_packages(),
    package_dir={'linearmodels': './linearmodels'},
    license='TBD/No License',
    author='Kevin Sheppard',
    url='https://gitlab.com/bashtage/linearmodels',
    long_description=open('README.md').read()
)
