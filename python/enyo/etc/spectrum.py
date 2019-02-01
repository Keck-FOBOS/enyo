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

from matplotlib import pyplot

#from mangadap.util.lineprofiles import FFTGaussianLSF
from mangadap.util.lineprofiles import IntegratedGaussianLSF

def spectral_coordinate_step(wave, log=False, base=10.0):
    """
    Return the sampling step for the input wavelength vector.

    If the sampling is logarithmic, return the change in the logarithm
    of the wavelength; otherwise, return the linear step in angstroms.

    Args: 
        wave (array-like):
            Wavelength coordinates of each spectral channel in
            angstroms.
        log (:obj:`bool`, optional):
            Input spectrum has been sampled geometrically.
        base (:obj:`float`, optional):
            If sampled geometrically, the sampling is done using a
            logarithm with this base.  For natural logarithm, use
            numpy.exp(1).

    Returns:
        :obj:`float`: Spectral sampling step in either angstroms
        (`log=False`) or the step in log(angstroms).

    Raises:
        ValueError:
            Raised if the wavelength vector is not uniformly (either
            linearly or log-linearly) sampled to numerical accuracy.
    """
    dw = numpy.diff(numpy.log(wave))/numpy.log(base) if log else numpy.diff(wave)
    if numpy.any( numpy.absolute(numpy.diff(dw)) > 100*numpy.finfo(dw.dtype).eps):
        raise ValueError('Wavelength vector is not uniformly sampled to numerical accuracy.')
    return numpy.mean(dw)


class Spectrum:
    r"""
    Define a spectrum.

    Units are expected to be wavelength in angstroms and flux in 1e-17
    erg/s/cm^2/angstrom.

    .. todo::
        - include inverse variance
        - include mask
        - incorporate astropy units?

    Args:
        wave (array-like):
            1D wavelength data in angstroms.  Expected to be sample
            linearly or geometrically.
        flux (array-like):
            1D flux data in 1e-17 erg/s/cm^2/angstrom.
        resolution (array-like, optional):
            1D spectral resolution (:math:`$R=\lambda/\Delta\lambda$`)
        log (:obj:`bool`, optional):
            Spectrum is sampled in steps of log base 10.
    """
    def __init__(self, wave, flux, resolution=None, log=False):
        if resolution is not None and len(flux) != len(resolution):
            raise ValueError('Resolution vector must match length of flux vector.')
        self.interpolator = interpolate.interp1d(wave, flux, assume_sorted=True)
        self.sres = resolution
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

    def show(self):
        pyplot.plot(self.wave, self.flux)
        pyplot.show()


class EmissionLineSpectrum(Spectrum):
    r"""
    Define an emission-line spectrum.

    .. todo::
        - use MaNGA DAP functions
        - apply surface-brightness dimming if redshift provided?
    
    Flux units are 1e-17 erg/s/cm^2/angstrom.

    Args:
        wave (array-like):
            1D wavelength data in angstroms.  Expected to be sampled
            linearly or geometrically.  These are the *observed*
            wavelengths.
        flux (array-like):
            The total fluxes of one or more emission lines in 1e-17 erg/s/cm^2.
        restwave (array-like):
            The central rest wavelengths of one or emission lines in angstroms.
        fwhm (array-like):
            The FWHM of the Gaussian line profiles.  If the resolution
            vector is provided, these are assumed to be the *intrinsic*
            widths, such that the line included in the spectrum has an
            observed with determined by that quadrature sum of the
            intrinsic and instrumental widths.  If the resolution vector
            is not provided, the line simply has the provided FWHM.
        units (:obj:`str`, optional):
            The units of the provided FWHM data.  Must be either 'km/s'
            or 'ang'.
        redshift (scalar-like, optional):
            If provided, the emission-line wavelengths are redshifted.
        continuum (array-like, optional):
            If provided, this is the 1D continuum placed below the line,
            which must have the same length as the input wavelength
            vector.  The continuum is 0 if not provided.
        resolution (array-like, optional):
            1D spectral resolution (:math:`$R=\lambda/\Delta\lambda$`)
        log (:obj:`bool`, optional):
            Spectrum is sampled in steps of log base 10.
    """
    def __init__(self, wave, flux, restwave, fwhm, units='ang', redshift=None, continuum=None,
                 resolution=None, log=False):
        # Check the input
        if units not in ['km/s', 'ang']:
            raise ValueError('FWHM units must be \'km/s\' or \'ang\'.')
        if resolution is not None and hasattr(resolution, '__len__') \
                and len(wave) != len(resolution):
            raise ValueError('Resolution vector must match length of wavelength vector.')
        if continuum is not None and len(wave) != len(continuum):
            raise ValueError('Continuum vector must match length of wavelength vector.')

        # Apply the redsift
        z = 0.0 if redshift is None else redshift

        # Set the line parameters
        _flux = numpy.asarray([flux]).ravel()
        _linewave = numpy.asarray([restwave]).ravel()*(1+z)
        sig2fwhm = numpy.sqrt(8.0 * numpy.log(2.0))
        sigma = numpy.asarray([fwhm]).ravel()/sig2fwhm
        nlines = len(_flux)

        # Check for consistency
        if len(_linewave) != nlines:
            raise ValueError('Number of line rest wavelengths must match number of line fluxes.')
        if len(sigma) != nlines:
            raise ValueError('Number of line FWHMs must match number of line fluxes.')

        # Convert the FWHM as needed based on the sampling and units
        if log and units == 'ang':
            # Convert to km/s
            sigma = astropy.constants.c.to('km/s').value*sigma/_linewave
        elif not log and units == 'km/s':
            # Convert to angstroms
            sigma = _linewave*sigma/astropy.constants.c.to('km/s').value

        # Include resolution if provided
        if resolution is not None:
            _resolution = resolution if hasattr(resolution, '__len__') \
                                else numpy.full_like(wave, resolution, dtype=float)
            sigma_inst = astropy.constants.c.to('km/s').value/_resolution/sig2fwhm if log else \
                            wave/_resolution/sig2fwhm
            interp = interpolate.interp1d(wave, sigma_inst, assume_sorted=True)
            sigma = numpy.sqrt(numpy.square(sigma) + numpy.square(interp(_linewave)))

        # Convert parameters to pixel units
        _dw = spectral_coordinate_step(wave, log=log)
        _linepix = (numpy.log10(_linewave) - numpy.log10(wave[0]) \
                        if log else _linewave - wave[0])/_dw

        # Flux to pixel units so that the spectrum has units of flux
        # density (flux per angstrom); less accurate when spectrum is
        # logarithmically binned
        dl = _linewave*(numpy.power(10.,_dw/2)-numpy.power(10.,-_dw/2)) if log else _dw
        _flux /= dl

        # Convert sigma to pixels
        sigma /= (astropy.constants.c.to('km/s').value*_dw*numpy.log(10.) if log else dl)
        
        # Construct the emission-line spectrum
        pix = numpy.arange(wave.size)
        spectrum = numpy.zeros(wave.size, dtype=float)
        profile = IntegratedGaussianLSF()
        for i in range(nlines):
            p = profile.parameters_from_moments(_flux[i], _linepix[i], sigma[i])
            spectrum += profile(pix, p)

        # Instantiate
        super(EmissionLineSpectrum, self).__init__(wave, spectrum, log=log)


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



