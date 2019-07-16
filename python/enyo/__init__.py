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
    import pkg_resources
    data_dir = pkg_resources.resource_filename('enyo', 'data')
    return os.path.split(data_dir) [0]

os.environ['ENYO_DIR'] = enyo_source_dir()

