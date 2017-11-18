import glob
import os
import sys

from setuptools import find_packages, setup

import versioneer

if sys.version_info < (3, 5):
    sys.exit('Requires Python 3.5 or later due to use of @ operator.')

try:
    markdown = os.stat('README.md').st_mtime
    if os.path.exists('README.rst'):
        rst = os.stat('README.rst').st_mtime
    else:
        rst = markdown - 1

    if rst >= markdown:
        with open('README.rst', 'r') as rst:
            description = rst.read()
    else:
        import pypandoc

        osx_line_ending = '\r'
        windows_line_ending = '\r\n'
        linux_line_ending = '\n'

        description = pypandoc.convert('README.md', 'rst')
        description = description.replace(windows_line_ending, linux_line_ending)
        description = description.replace(osx_line_ending, linux_line_ending)
        with open('README.rst', 'w') as rst:
            rst.write(description)

except (ImportError, OSError):
    import warnings
    warnings.warn("Unable to convert README.md to README.rst", UserWarning)
    description = open('README.md').read()

# Copy over notebooks from examples to docs for build
notebooks = glob.glob('examples/*.ipynb')
for nb in notebooks:
    fname = os.path.split(nb)[-1]
    folder, nbname = fname.split('_')
    outdir = os.path.join('doc', 'source', folder, 'examples')
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, nbname)
    with open(outfile, 'w') as nbout:
        with open(nb, 'r') as nbin:
            nbout.write(nbin.read())

images = glob.glob('examples/*.png')
for image in images:
    fname = os.path.split(image)[-1]
    folder, _ = fname.split('_')
    outdir = os.path.join('doc', 'source', folder, 'examples')
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, fname)
    with open(outfile, 'wb') as imageout:
        with open(image, 'rb') as imagein:
            imageout.write(imagein.read())

additional_files = []
for filename in glob.iglob('./linearmodels/datasets/**', recursive=True):
    if '.csv.bz' in filename:
        additional_files.append(filename.replace('./linearmodels/', ''))

for filename in glob.iglob('./linearmodels/tests/**', recursive=True):
    if '.txt' in filename:
        additional_files.append(filename.replace('./linearmodels/', ''))
    if '.csv' in filename:
        additional_files.append(filename.replace('./linearmodels/', ''))

for filename in glob.iglob('./examples/**', recursive=True):
    if '.png' in filename:
        additional_files.append(filename)

setup(
    cmdclass=versioneer.get_cmdclass(),
    name='linearmodels',
    license='NCSA',
    description='Instrumental Variable and Linear Panel models for Python',
    version=versioneer.get_version(),
    packages=find_packages(),
    package_dir={'linearmodels': './linearmodels'},
    author='Kevin Sheppard',
    author_email='kevin.k.sheppard@gmail.com',
    url='http://github.com/bashtage/linearmodels',
    long_description=description,
    install_requires=open('requirements.txt').read().split('\n'),
    include_package_data=True,
    package_data={'linearmodels': additional_files},
    keywords=['linear models', 'regression', 'instrumental variables', 'IV',
              'panel', 'fixed effects', 'clustered', 'heteroskedasticity',
              'endogeneity', 'instruments', 'statistics',
              'statistical inference', 'econometrics'],
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
)
