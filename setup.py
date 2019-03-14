from distutils.core import setup
from setuptools import find_packages

setup(name='MovieMaker',
      version='0.1',
      author='Andreas Kist',
      author_email='me@anki.xyz',
      packages=find_packages(),
      install_requires=['numpy', 'Pillow', 'matplotlib', 'subprocess', 'textwrap'])