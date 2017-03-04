from setuptools import setup, find_packages
import versioneer

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
