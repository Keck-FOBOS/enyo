#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Detector class
"""

import os
import numpy

from .efficiency import Efficiency

class Detector(Efficiency):
    """
    Define the detector statistics.

    Args:
        shape (:obj:`tuple`):
            Dimensions of the detector in number of pixels along the
            spectral axis and number of pixels along the spatial
            axis.
        pixelsize (:obj:`float`, optional):
            The size of the (square) detector pixels in *mm*.
        rn (:obj:`float`, optional):
            Read-noise in electrons.
        dark (:obj:`float`, optional):
            Dark current in electrons per second.
        gain (:obj:`float`, optional):
            Gain of detector amplifier in e- per ADU.
        fullwell (:obj:`float`, optional):
            The full well of the pixels in e-.
        nonlinear (:obj:`float`, optional):
            The fraction of the fullwell above which the detector
            response is nonlinear.
        qe (:obj:`float`, :class:`Efficiency`, optional):
            Detector quantum efficiency.
    """
    # TODO: Allow for multiple amplifiers per detector? Would also need
    # to define amplifier section.
    # TODO: Define overscan and data sections
    def __init__(self, shape, pixelsize=0.015, rn=1., dark=0., gain=1., fullwell=1e4,
                 nonlinear=1., qe=0.9):
        if len(shape) != 2:
            raise ValueError('Shape must contain two integers.')
        self.shape = shape
        self.pixelsize = pixelsize
        self.rn = rn
        self.dark = dark
        self.gain = gain
        self.fullwell = fullwell
        self.nonlinear = nonlinear
        if not isinstance(qe, (Efficiency, float)):
            raise TypeError('Provided quantum efficiency must be type `float` or `Efficiency`.')
        if isinstance(qe, float):
            super(Detector, self).__init__(qe)
        else:
            super(Detector, self).__init__(qe.eta, wave=qe.wave)

    def focalplane2pixel(self, x, y):
        """
        Convert a focal-plane position in mm to the detector pixel
        number.


# TODO: Define detector array


