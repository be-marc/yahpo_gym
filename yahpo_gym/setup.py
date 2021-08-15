import sys
from codecs import open
from os import path
from setuptools import (setup, find_packages)

here = path.abspath(path.dirname(__file__))

pypi_operations = frozenset(['register', 'upload']) & frozenset([x.lower() for x in sys.argv])
if pypi_operations:
    raise ValueError('Command(s) {} disabled in this example.'.format(', '.join(pypi_operations)))

with open(path.join(here, 'README.md'), encoding='utf-8') as fh:
    long_description = fh.read()

# __version__ = None
# exec(open('yahpo_gym/about.py').read())
# if __version__ is None:
#     raise IOError('about.py in project lacks __version__!')

setup(name='yahpo_gym',
      version='0.0.0',
      author='Florian Pfisterer',
      description='Inference module for the yahpo gym',
      long_description=long_description,
      license='LGPLv3',
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      include_package_data=True,
      install_requires=['ConfigSpace', 'onnxruntime'],
      keywords=['module', 'train', 'yahpo'],
      url="https://github.com/pfistfl/yahpo_gym",
      classifiers=["Development Status :: 3 - Alpha"])
