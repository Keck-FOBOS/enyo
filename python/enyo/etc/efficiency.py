#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Various efficiency calculations
"""

import os
import warnings
import inspect
import numpy

from matplotlib import pyplot

from scipy import interpolate

from astropy import units

class Efficiency:
    """
    Base class for efficiency data.

    Provided a wavelength independent efficiency value (`eta`), or a
    sampled vector of efficiency (`eta`) vs wavelength (`wave`), this
    class mainly just allows for access and interpolation of those
    data. When provided as a function of wavelength, the efficiency
    is assumed to be 0 outside the bounds of the provided wavelength
    vector.

    .. todo::
        - Allow efficiencies to be extrapolated?

    Args:
        eta (:obj:`float`, array-like):
            Constant or 1D efficiency data.
        wave (array-like, optional):
            1D array with wavelengths in angstroms.

    Attributes:
        interpolator (`scipy.interpolate.interp1d`):
            Linear interpolator used to sample efficiency at any
            wavelength, if vectors are provided.
    """
    def __init__(self, eta, wave=None):
        if wave is None:
            # eta has to be a constant
            if hasattr(eta, '__len__'):
                raise TypeError('If instantiated without a wavelength vector, the efficiency '
                                'must be a wavelength-independent contant.')
            self.interpolator = None
            self._eta = eta

        else:
            _wave = numpy.atleast_1d(wave)
            _eta = numpy.atleast_1d(eta)
            if _eta.ndim > 1 or len(_eta) == 1:
                raise ValueError('When providing wavelengths, efficiency must be a 1D vector '
                                 'with more than one element.')
            if _wave.shape != _eta.shape:
                raise ValueError('Efficiency and wavelengths must have the same shape.')
            self.interpolator = interpolate.interp1d(_wave, _eta, assume_sorted=True,
                                                     bounds_error=False, fill_value=0.0)
            self._eta = None
    
    @classmethod
    def from_file(cls, data_file, wave_units='angstrom'):
        """
        Read from an ascii file
        """
        if not os.path.isfile(data_file):
            raise FileNotFoundError('File does not exist: {0}'.format(data_file))
        db = numpy.genfromtxt(data_file)
        u = units.Unit(wave_units)
        return cls(db[:,1], wave=db[:,0]*u.to('angstrom'))

    def __call__(self, wave):
        _wave = numpy.atleast_1d(wave)
        if self.interpolator is None:
            return self._eta if _wave.size == 1 \
                        else numpy.full(_wave.shape, self._eta, dtype=float)
        _eta = self.interpolator(_wave)
        return _eta if hasattr(wave, '__len__') else _eta[0]

    def __getitem__(self, k):
        if self.interpolator is None:
            # TODO: Handle if k is a slice...
            warnings.warn('Efficiency is not a vector!  Returning constant value.')
            return self._eta
        return self.interpolator.y[k]

    @property
    def wave(self):
        if self.interpolator is None:
            warnings.warn('Efficiency is wavelength independent.')
            return None
        return self.interpolator.x

    @property
    def eta(self):
        if self.interpolator is None:
            return self._eta
        return self.interpolator.y

    def rescale(self, scale):
        """
        Scale must either be a single value or match the size of the
        existing eta vector.
        """
        if self.interpolator is None:
            self._eta *= scale
        else:
            self.interpolator.y *= scale


class CombinedEfficiency(Efficiency):
    """
    A class that combines multiple efficiencies that can be accessed
    separately or act as a single efficiency.

    Args:
        efficiencies (:obj:`list`, :obj:`dict`):
            The set of efficiencies to combine. Nominally this should
            be a dictionary that gives the efficiencies and a keyword
            identifier for each. A list can be entered, meaning that
            the efficiencies can only be access by their index, not a
            keyword.
        wave (array-like):
            Wavelengths of/for efficiency measurements.

    Attributes:
        efficiencies (:obj:`dict`):
            The efficiencies combined. Access to individual
            efficiencies is by keyword; if keywords not provided,
            access is by single integer index.
    """
    def __init__(self, efficiencies, wave=None):
        if isinstance(efficiencies, list):
            self.efficiencies = dict([(i,eff) for i,eff in enumerate(efficiencies)])
        elif isinstance(efficiencies, dict):
            self.efficiencies = efficiencies
        else:
            raise TypeError('Efficiencies to include must provided as a list or dict.')

        # Make sure the components of self are Efficiency objects
        for eff in self.efficiencies.values():
            if not isinstance(eff, Efficiency):
                raise TypeError('Each element of input must be an Efficiency object.')

        if wave is None:
            # Consolidate wavelengths from other efficiencies
            wave = numpy.empty(0, dtype=float)
            for inp in self.efficiencies.values():
                if inp.wave is None:
                    continue
                wave = numpy.append(wave, inp.wave)
            wave = None if len(wave) == 0 else numpy.sort(wave)
            if wave is None:
                warnings.warn('No wavelengths provided for any efficiencies to combine.')

        # Construct the total efficiency
        total = 1. if wave is None else numpy.ones_like(wave, dtype=float)
        for eff in self.efficiencies.values():
            total *= (eff.eta if wave is None else eff(wave))

        super(CombinedEfficiency, self).__init__(total, wave=wave)

    @classmethod
    def from_total(cls, total, wave=None):
        return cls({'total': Efficiency(total, wave=wave)})

    def keys(self):
        return self.efficiencies.keys()

    def __getitem__(self, key):
        """Return the specified efficiency."""
        return self.efficiencies[key]


class FiberThroughput(Efficiency):
    def __init__(self, fiber='polymicro'):
        data_file = FiberThroughput.select_data_file(fiber)
        if not os.path.isfile(data_file):
            raise FileNotFoundError('No file: {0}'.format(data_file))
        db = numpy.genfromtxt(data_file)
        super(FiberThroughput, self).__init__(db[:,1], wave=db[:,0])

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
        super(FilterResponse, self).__init__(db[:,1], wave=db[:,0])


class AtmosphericThroughput(Efficiency):
    def __init__(self, airmass=1.0, location='maunakea'):
        if location == 'maunakea':
            db = numpy.genfromtxt(os.path.join(os.environ['ENYO_DIR'], 'data', 'sky',
                                  'mauna_kea_extinction.db'))
        else:
            raise NotImplementedError('Extinction unknown at {0}.'.format(location))
        self.airmass = airmass
        super(AtmosphericThroughput, self).__init__(numpy.power(10, -0.4*db[:,1]*self.airmass),
                                                    wave=db[:,0])

    def reset_airmass(self, airmass):
        self.rescale(numpy.power(self.interpolator.y, (airmass/self.airmass -1)))


class SpectrographThroughput(CombinedEfficiency):
    """
    Define the system throughput from the telescope focal plane to the detector.
    """
    def __init__(self, wave=None, coupling=None, fibers=None, grating=None, camera=None,
                 detector=None, other=None):
        values = inspect.getargvalues(inspect.currentframe())
        keys = numpy.array(values.args[2:])
        objects = numpy.array([values.locals[key] for key in keys])
        indx = numpy.array([o is not None for o in objects])
        efficiencies = dict([(k,o) for k,o in zip(keys[indx], objects[indx])])
        super(SpectrographThroughput, self).__init__(efficiencies, wave=wave)


class SystemThroughput(CombinedEfficiency):
    """
    Define the system throughput from the top of the telescope to the detector.
    """
    # TODO: Also allow for 'other' here?
    def __init__(self, wave=None, spectrograph=None, telescope=None):
        values = inspect.getargvalues(inspect.currentframe())
        keys = numpy.array(values.args[2:])
        objects = numpy.array([values.locals[key] for key in keys])
        indx = numpy.array([o is not None for o in objects])
        efficiencies = dict([(k,o) for k,o in zip(keys[indx], objects[indx])])
        super(SystemThroughput, self).__init__(efficiencies, wave=wave)
