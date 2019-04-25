#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Various efficiency calculations
"""

import os
import numpy

from matplotlib import pyplot

from scipy import interpolate

from astropy import units

from . import source

class Efficiency:
    """
    Base class for efficiency data.

    .. todo::
        - Allow for x,y detector pixel to convert to wavelength and
          pseudo-slit/slit position; field angle fixed for fiber
          pseudo-slit, not for imaging spectrograph machined slit
        - Allow for x,y focal plane position to convert to detector
          pixel

    Args:
        wave (array-like):
            1D array with wavelengths in angstroms.
        eta (array-like):
            1D array with efficiency data.
    """
    def __init__(self, wave, eta):
        self.interpolator = interpolate.interp1d(wave, eta, assume_sorted=True, bounds_error=False,
                                                 fill_value=0.0)
    
    @classmethod
    def from_file(cls, data_file, wave_units='angstrom'):
        """
        Read from an ascii file
        """
        db = numpy.genfromtxt(data_file)
        u = units.Unit(wave_units)
        return cls(db[:,0]*u.to('angstrom'), db[:,1])

    def __call__(self, wave):
        return self.interpolator(wave)

    def __getitem__(self, k):
        return self.interpolator.y[k]

    @property
    def wave(self):
        return self.interpolator.x

    @property
    def eta(self):
        return self.interpolator.y


class FiberThroughput(Efficiency):
    def __init__(self, fiber='polymicro'):
        data_file = FiberThroughput.select_data_file(fiber)
        if not os.path.isfile(data_file):
            raise FileNotFoundError('No file: {0}'.format(data_file))
        db = numpy.genfromtxt(data_file)
        super(FiberThroughput, self).__init__(db[:,0], db[:,1])

    @staticmethod
    def select_data_file(fiber):
        if fiber == 'polymicro':
            return os.path.join(os.environ['ENYO_DIR'], 'data', 'efficiency', 'fibers',
                                'polymicro.db')
        raise NotImplementedError('Unknown fiber type: {0}'.format(fiber))


class FilterResponse(Efficiency):
    """
    The efficiency of a broad-band filter.

    Args:
        band (:obj:`str`, optional):
            The band to use.  Options are for the response functions in
            the data/broadband_filters directory.

    Raises:
        FileNotFoundError:
            Raised if the default file for the given band is not
            available.
    """
    def __init__(self, band='g'):
        data_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'broadband_filters',
                                 'gunn_2001_{0}_response.db'.format(band))
        if not os.path.isfile(data_file):
            raise FileNotFoundError('No file: {0}'.format(data_file))
        db = numpy.genfromtxt(data_file)
        super(FilterResponse, self).__init__(db[:,0], db[:,1])


class SpectrographThroughput(Efficiency):
    """
    Define the system throughput from the telescope focal plane to the detector.
    """
    def __init__(self, wave, detector=None, camera=None, grating=None, fibers=None,
                 coupling=None, total=None):
        self.detector = detector
        self.camera = camera
        self.grating = grating
        self.fibers = fibers
        self.coupling = coupling
        if total is None:
            self.total = None
            self._get_total()
        else:
            self.total = total
        super(SpectrographThroughput, self).__init__(wave, self.total)
        
    def _get_total(self, wave):
        if self.total is not None:
            raise ValueError('Spectrograph efficiency already calculated!')

        # Initialize to unity
        self.total = numpy.ones_like(wave, dtype=float)
        components = [self.detector, self.camera, self.grating, self.fibers, self.coupling]
        if all(a is None for a in components):
            warnings.warn('None of the relevant efficiencies are defined for the spectrograph '
                          'throughput.  Will return unity!')
            return

        for c in components:
            if c is None:
                continue
            self.total *= c(wave)
    

class SystemThroughput(Efficiency):
    """
    Define the system throughput from the top of the telescope to the detector.
    """
    def __init__(self, wave, spectrograph=None, surfaces=None, coating=None, total=None):
        self.spectrograph = spectrograph
        self.surfaces = surfaces
        self.coating = coating
        if total is None:
            self.total = None
            self._get_total(wave)
        else:
            self.total = total
        super(SystemThroughput, self).__init__(wave, self.total)

    def _get_total(self, wave):
        if self.total is not None:
            raise ValueError('System efficiency already calculated!')

        # Initialize to unity
        self.total = numpy.ones_like(wave, dtype=float)
        components = [self.spectrograph, self.coating]
        if all(a is None for a in components):
            warnings.warn('None of the relevant efficiencies are defined for the system '
                          'throughput.  Will return unity!')
            return

        if self.spectrograph is not None:
            self.total *= self.spectrograph(wave)
        if self.coating is not None:
            if self.surfaces is None:
                self.surfaces = 1           # Assume prime focus
            self.total *= numpy.power(self.coating(wave), self.surfaces)


class AtmosphericThroughput(Efficiency):
    def __init__(self, airmass=1.0, location='maunakea'):
        if location != 'maunakea':
            raise NotImplementedError('Extinction unknown at {0}.'.format(location))
        db = numpy.genfromtxt(os.path.join(os.environ['ENYO_DIR'], 'data/sky',
                               'mauna_kea_extinction.db'))
        super(AtmosphericThroughput, self).__init__(db[:,0], numpy.power(10, -0.4*db[:,1]*airmass))


class Detector:
    """
    Define the detector statistics.

    .. todo:
        - why isn't efficiency the base class for this?
        - allow pixel and dispersion scale to be wavelength dependent
        - set pixel size in micron; set pixel scale using telescope
          plate scale
        - set dispersion scale using pixel size? need grating, etc...

    Args:
        pixelscale (:obj:`float`, optional):
            The pixel scale in arcseconds per pixel.
        dispscale (:obj:`float`, optional):
            The dispersion scale in angstroms per pixel.
        log (:obj:`bool`, optional):
            The dispersion scale is in log(angstroms) per pixel.
        rn (:obj:`float`, optional):
            Read-noise in electrons.
        dark (:obj:`float`, optional):
            Dark current in electrons per second.
        qe (:obj:`float`, :class:`Efficiency`, optional):
            Detector quantum efficiency.
    """
    def __init__(self, pixelscale=1., dispscale=1., log=False, rn=1., dark=0., qe=0.9):
        self.pixelscale = pixelscale
        self.dispscale = dispscale
        self.log = log
        self.rn = rn
        self.dark = dark
        self.qe = qe

    def efficiency(self, wave=None):
        if isinstance(self.qe, Efficiency):
            if wave is None:
                raise ValueError('Must provide wavelength for quantum efficiency')
            return self.qe(wave)
        elif not hasattr(wave, '__len__'):
            return self.qe
        return self.qe if wave is None else numpy.full_like(wave, self.qe, dtype=float)


