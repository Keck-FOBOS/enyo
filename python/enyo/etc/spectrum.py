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


def convert_flux_density(wave, flux, density='ang'):
    r"""
    Convert a spectrum with flux per unit wavelength to per unit
    frequency or vice versa.

    For converting from per unit wavelength, this function returns
    
    .. math::
        
        F_{\nu} = F_{\lambda} \frac{d\lambda}{d\nu} = F_{\lambda}
        \frac{\lambda^2}{c}.

    The spectrum independent variable (`wave`) is always expected to be
    the wavelength in angstroms.  The input/output units always expect
    :math:`F_{\lambda}` in :math:`10^{-17}\ {\rm erg\ s}^{-1}\ {\rm
    cm}^{-2}\ {\rm A}^{-1}` and :math:`F_{\nu}` in microjanskys
    (:math:`10^{-29} {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ {\rm Hz}^{-1}`).
    Beyond this, the function is ignorant of the input/output units.

    E.g., if you provide the function with an input spectrum with
    :math:`F_{\lambda}` in :math:`10^{-11}\ {\rm erg\ s}^{-1}\ {\rm
    cm}^{-2}\ {\rm A}^{-1}`, the output will be :math:`F_{\nu}` in
    Janskys.

    Args:
        wave (:obj:`float`, array-like):
            The vector with the wavelengths in angstroms.
        flux (:obj:`float`, array-like):
            The vector with the flux density; cf. `density`.
        density (:obj:`str`, optional):
            The density unit of the *input* spectrum.  Must be either
            'ang' or 'Hz'.  If the input spectrum is :math:`F_{\lambda}`
            (`density='ang'`), the returned spectrum is :math:`F_{\nu}`.

    Returns:
        The spectrum with the converted units.

    Raises:
        ValueError:
            Raised if the `wave` and `flux` arguments do not have the
            same shape.
    """
    # Set to be at least vectors
    _wave = numpy.atleast_1d(wave)
    _flux = numpy.atleast_1d(flux)
    if _wave.shape != _flux.shape:
        raise ValueError('Wavelength and flux arrays must have the same shape.')
    if density == 'ang':
        # Convert Flambda to Fnu
        fnu = _flux*numpy.square(_wave)*1e12/astropy.constants.c.to('angstrom/s').value
        return fnu[0] if isinstance(flux, float) else fnu
    if density == 'Hz':
        # Convert Fnu to Flambda
        flambda = _flux*astropy.constants.c.to('angstrom/s').value/numpy.square(_wave)/1e12
        return flambda[0] if isinstance(flux, float) else flambda
    raise ValueError('Density units must be either \'ang\' or \'Hz\'.')


class Spectrum:
    r"""
    Define a spectrum.

    Units are expected to be wavelength in angstroms and flux in 1e-17
    erg/s/cm^2/angstrom.

    .. todo::
        - include inverse variance
        - include mask
        - incorporate astropy units?
        - keep track of units

    Args:
        wave (array-like):
            1D wavelength data in angstroms.  Expected to be sampled
            linearly or geometrically.
        flux (array-like):
            1D flux data in 1e-17 erg/s/cm^2/angstrom.
        resolution (array-like, optional):
            1D spectral resolution (:math:`$R=\lambda/\Delta\lambda$`)
        log (:obj:`bool`, optional):
            Spectrum is sampled in steps of log base 10.
    """
    def __init__(self, wave, flux, error=None, resolution=None, log=False):
        if resolution is not None and len(flux) != len(resolution):
            raise ValueError('Resolution vector must match length of flux vector.')
        self.interpolator = interpolate.interp1d(wave, flux, assume_sorted=True)
        self.error = error
        self.sres = resolution
        self.log = log
        # TODO: Check log against the input wavelength vector
        self.nu = self._frequency()
        self.fnu = None

    @property
    def wave(self):
        return self.interpolator.x

    @property
    def flux(self):
        return self.interpolator.y

    def __getitem__(self, s):
        return self.interpolator.y[s]

    def _frequency(self):
        """Calculate the frequency in terahertz."""
        return 10*astropy.constants.c.to('km/s').value/self.wave

    def interp(self, w):
        if not isinstance(w, numpy.ndarray) and w > self.interpolator.x[0] \
                and w < self.interpolator.x[-1]:
            return self.interpolator(w)

        indx = (w > self.interpolator.x[0]) & (w < self.interpolator.x[-1])
        sampled = numpy.zeros_like(w, dtype=float)
        sampled[indx] = self.interpolator(w[indx])
        return sampled

    @classmethod
    def from_file(cls, fitsfile, waveext='WAVE', fluxext='FLUX', resext=None):
        hdu = fits.open(fitsfile)
        wave = hdu[waveext].data
        flux = hdu[fluxext].data
        sres = None if resext is None else hdu[resext].data
        return cls(wave, flux, resolution=sres)

    def wavelength_step(self):
        """
        Return the wavelength step per pixel.
        """
        # TODO: Lazy load and then keep this?
        dw = spectral_coordinate_step(self.wave, log=self.log)
        if self.log:
            dw *= numpy.log(10.)*self.wave
        return dw

    def frequency_step(self):
        """
        Return the frequency step per pixel in THz.
        """
        return 10*astropy.constants.c.to('km/s').value*self.wavelength_step()/self.wave/self.wave

    def magnitude(self, band, system='AB'):
        if system == 'AB':
            if self.nu is None:
                self._frequency()
            if self.fnu is None:
                # Flux in microJanskys
                self.fnu = convert_flux_density(self.wave, self.flux)
            dnu = self.frequency_step() 
            band_weighted_mean = numpy.sum(band(self.wave)*self.fnu*dnu) \
                                    / numpy.sum(band(self.wave)*dnu)
            return -2.5*numpy.log10(band_weighted_mean*1e-29) - 48.6

        raise NotImplementedError('Photometric system {0} not implemented.'.format(system))

    def rescale(self, factor):
        """
        Rescale the spectrum by the provided factor.

        The spectral data are modified *in-place*; nothing is
        returned.

        Args:
            factor (scalar-like or array-like):
                Factor to multiply the fluxes. If it's a vector, it
                must have the same length as the existing spectrum.
        """
        self.interpolator.y *= factor
        if self.error is not None:
            # Adjust the error
            self.error *= numpy.absolute(factor)
        if self.fnu is not None:
            # Adjust the flux density per Hz
            self.fnu *= factor

    def rescale_flux(self, wave, flux):
        """
        Rescale the spectrum to be specifically the flux value at the
        provided wavelength. The input flux should be in units of
        1e-17 erg/s/cm^2/angstrom.
        """
        return self.rescale(flux/self.interp(wave))

    def rescale_magnitude(self, band, new_mag, system='AB'):
        """
        Rescale existing magnitude to a new magnitude.
        """
        dmag = new_mag - self.magnitude(band, system=system)
        if system == 'AB':
            return self.rescale(numpy.power(10., -dmag/2.5))
        raise NotImplementedError('Photometric system {0} not implemented.'.format(system))

    def photon_flux(self):
        r"""
        Convert the spectrum from 1e-17 erg/s/cm^2/angstrom to
        photons/s/cm^2/angstrom.

        The spectral data are modified *in-place*; nothing is
        returned.
        """
        ergs_per_photon = astropy.constants.h.to('erg s') * astropy.constants.c.to('angstrom/s') \
                            / (self.wave * astropy.units.angstrom)
        return self.rescale(1e-17 / ergs_per_photon.value)

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

        indx = (_linewave > wave[0]) & (_linewave < wave[-1])
        if not numpy.any(indx):
            raise ValueError('Redshifted lines are all outside of the provided wavelength range.')

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
            sigma[indx] = numpy.sqrt(numpy.square(sigma[indx])
                                        + numpy.square(interp(_linewave[indx])))

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
            if not indx[i]:
                continue
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
        raise NotImplementedError('Maunakea sky spectrum is fixed.')


class ABReferenceSpectrum(Spectrum):
    """
    Construct a spectrum with a constant flux of 3631 Jy.

    Inherits from :class:`Spectrum`, which we take to mean that the flux
    is always in units of 1e-17 erg/s/cm^2/angstrom.
    """
    def __init__(self, wave, log=False):
        norm = numpy.power(10., 29 - 48.6/2.5)  # Reference flux in microJanskys
        fnu = numpy.full_like(wave, norm, dtype=float)
        flambda = convert_flux_density(wave, fnu, density='Hz')
        super(ABReferenceSpectrum, self).__init__(wave, flambda, log=log)

