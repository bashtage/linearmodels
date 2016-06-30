from setuptools import setup, find_packages

setup(
    name='panel',
    version='0.0.1',
    packages=find_packages(),
    package_dir={'panel': './panel'},
    license='TBD/No License',
    author='Kevin Sheppard',
    url='https://gitlab.com/bashtage/panel',
    long_description=open('README.md').read()
)
