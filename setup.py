from setuptools import setup, find_packages
import versioneer

setup(
    cmdclass=versioneer.get_cmdclass(),
    name='panel',
    version=versioneer.get_version(),
    packages=find_packages(),
    package_dir={'panel': './panel'},
    license='TBD/No License',
    author='Kevin Sheppard',
    url='https://gitlab.com/bashtage/panel',
    long_description=open('README.md').read()
)
