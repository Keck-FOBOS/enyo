#!/bin/env/python3
# -*- encoding: utf-8 -*-

import os

__version__ = '0.1.0dev'
__license__ = 'BSD3'
__author__ = 'Kyle B. Westfall'
__maintainer__ = 'Kyle B. Westfall'
__email__ = 'westfall@ucolick.org'
__copyright__ = '(c) 2018, Kyle B. Westfall'

def enyo_source_dir():
    """Return the root path to the DAP source directory."""
    dirlist = os.path.dirname(os.path.abspath(__file__)).split('/')[:-2]
    return os.path.join(os.sep, *dirlist) if dirlist[0] == '' else os.path.join(*dirlist)

os.environ['ENYO_DIR'] = enyo_source_dir()

