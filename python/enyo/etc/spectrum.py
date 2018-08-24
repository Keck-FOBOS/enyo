#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Spectrum utilities
"""

import os
import numpy
from scipy import interpolate
from astropy.io import fits
import astropy.constants
import astropy.units

def spectral_coordinate_step(wave, log=False, base=10.0):
    """
    Return the sampling step for the input wavelength vector.

    If the sampling is logarithmic, return the change in the logarithm
    of the wavelength; otherwise, return the linear step in angstroms.

    Args: 
        wave (numpy.ndarray): Wavelength coordinates of each spectral
            channel in angstroms.
        log (bool): (**Optional**) Input spectrum has been sampled
            geometrically.
        base (float): (**Optional**) If sampled geometrically, the
            sampling is done using a logarithm with this base.  For
            natural logarithm, use numpy.exp(1).

    Returns:
        float: Spectral sampling step in either angstroms (log=False) or
        the step in log(angstroms).
    """
    dw = numpy.diff(numpy.log(wave))/numpy.log(base) if log else numpy.diff(wave)
    if numpy.any( numpy.absolute(numpy.diff(dw)) > 100*numpy.finfo(dw.dtype).eps):
        raise ValueError('Wavelength vector is not uniformly sampled to numerical accuracy.')
    return numpy.mean(dw)


class FilterResponse:
    def __init__(self, band='g'):
        data_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'broadband_filters',
                                 'gunn_2001_{0}_response.db'.format(band))
        if not os.path.isfile(data_file):
            raise FileNotFoundError('No file: {0}'.format(data_file))

        self.response_func = numpy.genfromtxt(data_file)
        self.interpolator = interpolate.interp1d(self.response_func[:,0], self.response_func[:,1],
                                                 bounds_error=False, fill_value=0.0,
                                                 assume_sorted=True)
    def __call__(self, wave):
        return self.interpolator(wave)

class Spectrum:
    """
    Define a spectrum.
    
    Units are expected to be wavelengt in angstroms and flux in 1e-17
    erg/s/cm^2/angstrom.
    """
    def __init__(self, wave, flux, resolution=None, log=False):
        self.sres = resolution
        self.interpolator = interpolate.interp1d(wave, flux, assume_sorted=True)
        self.log = log

    @property
    def wave(self):
        return self.interpolator.x

    @property
    def flux(self):
        return self.interpolator.y

    def __getitem__(self, s):
        return self.interpolator.y[s]

    def interp(self, w):
        return self.interpolator(w)

    @classmethod
    def from_file(cls, fitsfile, waveext='WAVE', fluxext='FLUX', resext=None):
        hdu = fits.open(fitsfile)
        wave = hdu[waveext].data
        flux = hdu[fluxext].data
        sres = None if resext is None else hdu[resext].data
        return cls(wave, flux, resolution=sres)

    def wavelength_step(self):
        dw = spectral_coordinate_step(self.wave, log=self.log)
        if self.log:
            dw *= numpy.log(10.)*self.wave
        return dw

    def rescale_flux(self, wave, flux):
        """
        input flux should be in units of  1e-17 erg/s/cm^2/angstrom.
        """
        self.interpolator.y *= flux/self.interp(wave)

    def rescale_magnitude(self, band, new_mag, system='AB'):
        # Get the current magnitude
        dw = self.wavelength_step()
        band_weighted_mean = numpy.sum(band(self.wave)*self.flux*dw) \
                                / numpy.sum(band(self.wave)*dw)
        band_weighted_center = numpy.sum(band(self.wave)*self.wave*self.flux*dw) \
                                / numpy.sum(band(self.wave)*self.flux*dw)
        # units are 1e-17 erg/s/cm^2/angstrom
        if system == 'AB':
            current_mag = -2.5*numpy.log10(3.34e4*numpy.square(band_weighted_center)
                            * 1e-17*band_weighted_mean) + 8.9
            self.interpolator.y *= numpy.power(10, -0.4*(new_mag-current_mag))
            return

        raise NotImplementedError('Photometric system {0} not implemented.'.format(system))

    def photon_flux(self):
        """
        Convert the spectrum from erg/s/cm^2/angstrom to photons/s/cm^2/angstrom
        """
        ergs_per_photon = astropy.constants.h.to('erg s') * astropy.constants.c.to('angstrom/s') \
                            / (self.wave * astropy.units.angstrom)
        return 1e-17*self.interpolator.y / ergs_per_photon.value


# 8329-6104
class BlueGalaxySpectrum(Spectrum):
    def __init__(self, redshift=0.0):
        fitsfile = os.path.join(os.environ['ENYO_DIR'], 'data/galaxy/blue_galaxy_8329-6104.fits')
        hdu = fits.open(fitsfile)
        wave = hdu['WAVE'].data * (1+redshift)
        flux = hdu['FLUX'].data
        super(BlueGalaxySpectrum, self).__init__(wave, flux, log=True)

    @classmethod
    def from_file(cls):
        raise NotImplementedError('Spectrum for blue galaxy is fixed.')

# 8131-6102
class RedGalaxySpectrum(Spectrum):
    def __init__(self, redshift=0.0):
        fitsfile = os.path.join(os.environ['ENYO_DIR'], 'data/galaxy/red_galaxy_8131-6102.fits')
        hdu = fits.open(fitsfile)
        wave = hdu['WAVE'].data * (1+redshift)
        flux = hdu['FLUX'].data
        super(RedGalaxySpectrum, self).__init__(wave, flux)

    @classmethod
    def from_file(cls):
        raise NotImplementedError('Spectrum for red galaxy is fixed.')

class MaunakeaSkySpectrum(Spectrum):
    def __init__(self):
        fitsfile = os.path.join(os.environ['ENYO_DIR'], 'data/sky/manga/apo2maunakeasky.fits')
        hdu = fits.open(fitsfile)
        wave = hdu['WAVE'].data
        flux = hdu['FLUX'].data
        super(MaunakeaSkySpectrum, self).__init__(wave, flux, log=True)

    @classmethod
    def from_file(cls):
        raise NotImplementedError('Spectrum for blue galaxy is fixed.')



