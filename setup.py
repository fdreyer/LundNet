# This file is part of LundNet by F. Dreyer and H. Qu

from __future__ import print_function
import sys
from setuptools import setup, find_packages

if sys.version_info < (3,6):
    print("LundNet requires Python 3.6 or later", file=sys.stderr)
    sys.exit(1)

with open('README.md') as f:
    long_desc = f.read()

setup(name= "lundnet",
      version = '1.0.0',
      description = "A jet tagging algorithm based on graph networks",
      author = "F. Dreyer, H. Qu",
      author_email = "frederic.dreyer@cern.ch, huilin.qu@cern.ch",
      url="https://github.com/fdreyer/lundnet",
      long_description = long_desc,
      entry_points = {'console_scripts':
                      ['lundnet = lundnet.scripts.lundnet:main']},
      package_dir = {'': 'src'},
      packages = find_packages('src'),
      zip_safe = False,
      classifiers=[
            'Operating System :: Unix',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            ],
     )

