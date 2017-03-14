import glob
import os

from setuptools import setup, find_packages

import versioneer

with open('requirements.txt', 'r') as req:
    requirements = req.read().split('\n')

# Copy over notebooks from examples to docs for build
notebooks = glob.glob('examples/*.ipynb')
for nb in notebooks:
    fname = os.path.split(nb)[-1]
    folder, nbname = fname.split('_')
    outdir = outfile = os.path.join('doc', 'source', folder, 'examples')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outfile = os.path.join(outdir, nbname)
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
    long_description=open('README.md').read(),
    install_requires=requirements
)
