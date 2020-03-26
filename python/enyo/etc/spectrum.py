#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Spectrum utilities
"""

import os

from IPython import embed

import numpy
from scipy import interpolate
from astropy.io import fits
import astropy.constants
import astropy.units

from matplotlib import pyplot

from ..util.lineprofiles import IntegratedGaussianLSF
from .sampling import Resample

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


def angstroms_per_pixel(wave, log=False, base=10.0, regular=True):
    """
    Return a vector with the angstroms per pixel at each channel.

    When `regular=True`, the function assumes that the wavelengths are
    either sampled linearly or geometrically.  Otherwise, it calculates
    the size of each pixel as the difference between the wavelength
    coordinates.  The first and last pixels are assumed to have a width
    as determined by assuming the coordinate is at its center.

    .. note::

        If the regular is False and log is True, the code does *not*
        assume the wavelength coordinates are at the geometric center of
        the pixel.

    Args:
        wave (`numpy.ndarray`_):
            (Geometric) centers of the spectrum pixels in angstroms.
        log (`numpy.ndarray`_, optional):
            The vector is geometrically sampled.
        base (:obj:`float`, optional):
            Base of the logarithm used in the geometric sampling.
        regular (:obj:`bool`, optional):
            The vector is regularly sampled.

    Returns:
        numpy.ndarray: The angstroms per pixel.
    """
    if regular:
        ang_per_pix = spectral_coordinate_step(wave, log=log, base=base)
        return ang_per_pix*wave*numpy.log(base) if log else numpy.repeat(ang_per_pix, len(wave))

    return numpy.diff([(3*wave[0]-wave[1])/2] + ((wave[1:] + wave[:-1])/2).tolist()
                      + [(3*wave[-1]-wave[-2])/2])


def convert_flux_density(wave, flux, density='ang'):
    r"""
    Convert a spectrum with flux per unit wavelength to per unit
    frequency or vice versa.

    For converting from per unit wavelength, this function returns
    
    .. math::
        
        F_{\nu} = F_{\lambda} \frac{d\lambda}{d\nu} = F_{\lambda}
        \frac{\lambda^2}{c}.

    The spectrum independent variable (`wave`) is always expected to
    be the wavelength in angstroms. The input/output units always
    expect :math:`F_{\lambda}` in :math:`10^{-17}\ {\rm erg\ s}^{-1}\
    {\rm cm}^{-2}\ {\rm A}^{-1}` and :math:`F_{\nu}` in microjanskys
    (:math:`10^{-29} {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ {\rm
    Hz}^{-1}`). Beyond this, the function is ignorant of the
    input/output units. For example, if you provide the function with
    an input spectrum with :math:`F_{\lambda}` in :math:`10^{-11}\
    {\rm erg\ s}^{-1}\ {\rm cm}^{-2}\ {\rm A}^{-1}`, the output will
    be :math:`F_{\nu}` in Janskys.

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
        - include mask
        - incorporate astropy units?
        - keep track of units
        - allow wavelength vector to be irregularly gridded

    Args:
        wave (array-like):
            1D wavelength data in angstroms.  Expected to be sampled
            linearly or geometrically.
        flux (array-like):
            1D flux data in 1e-17 erg/s/cm^2/angstrom.
        resolution (float, array-like, optional):
            1D spectral resolution (:math:`$R=\lambda/\Delta\lambda$`)
        log (:obj:`bool`, optional):
            Spectrum is sampled in steps of log base 10.
    """
    def __init__(self, wave, flux, error=None, resolution=None, log=False):
        # Check the input
        _wave = numpy.atleast_1d(wave)
        _flux = numpy.atleast_1d(flux)
        if _wave.ndim != 1:
            raise ValueError('Spectrum can only accommodate single vectors for now.')
        if _wave.shape != _flux.shape:
            raise ValueError('Wavelength and flux vectors do not match.')

        self.sres = None if resolution is None else numpy.atleast_1d(resolution)
        if self.sres is not None:
            # Allow resolution to be a single value
            if self.sres.size == 1:
                self.sres = numpy.repeat(self.sres, _flux.size)
            if self.sres.shape != _flux.shape:
                raise ValueError('Resolution vector must match length of flux vector.')
            
        self.error = None if error is None else numpy.atleast_1d(error)
        if self.error is not None and self.error.shape != _flux.shape:
            raise ValueError('Error vector must match length of flux vector.')
            
        self.interpolator = interpolate.interp1d(_wave, _flux, assume_sorted=True)
        self.size = _wave.size
        # TODO: Check log against the input wavelength vector
        self.log = log
        self.nu = self._frequency()
        self.fnu = None

    @property
    def wave(self):
        """
        The wavelength data vector.
        """
        return self.interpolator.x

    @property
    def flux(self):
        """
        The flux data vector.
        """
        return self.interpolator.y

    def copy(self):
        return Spectrum(self.wave.copy(), self.flux.copy(),
                        error=None if self.error is None else self.error.copy(),
                        resolution=None if self.sres is None else self.sres.copy(), log=self.log)

    def _arith(self, other, func):
        if isinstance(other, Spectrum):
            return func(other)

        _other = numpy.atleast_1d(other)
        if _other.size == 1:
            _other = numpy.repeat(_other, self.size)
        if _other.size != self.size:
            raise ValueError('Vector to add has incorrect length.')
        return func(Spectrum(self.interpolator.x, _other))

    def __add__(self, rhs):
        return self._arith(rhs, self._add_spectrum)

    def __radd__(self, lhs):
        return self + lhs

    def __sub__(self, rhs):
        return self + rhs*-1

    def __rsub__(self, lhs):
        return self*-1 + lhs

    def __mul__(self, rhs):
        return self._arith(rhs, self._mul_spectrum)

    def __rmul__(self, lhs):
        return self * lhs

    def __truediv__(self, rhs):
        if isinstance(rhs, Spectrum):
            return self * rhs.inverse()
        # NOTE: Parentheses are important below because they force the
        # call to __mul__ for the correct quantity and avoid an
        # infinite loop!
        return self * (1/float(rhs))

    def __rtruediv__(self, lhs):
        return self.inverse() * lhs

    def inverse(self):
        error = None if self.error is None else self.error/numpy.square(self.interpolator.y)
        return Spectrum(self.wave, 1./self.interpolator.y, error=error, resolution=self.sres,
                        log=self.log)

    def _mul_spectrum(self, rhs):
        """
        rhs must have type Spectrum
        """
        if not numpy.array_equal(rhs.wave, self.wave):
            raise NotImplementedError('To perform arithmetic on spectra, their wavelength arrays '
                                      'must be identical.')
        flux = self.flux * rhs.flux
        if self.error is None and rhs.error is None:
            error = None
        else:
            error = numpy.zeros(self.size, dtype=float)
            if self.error is not None:
                error += numpy.square(self.error/self.flux)
            if rhs.error is not None:
                error += numpy.square(rhs.error/rhs.flux)
            error = numpy.sqrt(error) * numpy.absolute(flux)
        if self.sres is not None and rhs.sres is not None \
                and not numpy.array_equal(self.sres, rhs.sres):
            warnings.warn('Spectral resolution is not correctly propagated.')
        sres = self.sres if self.sres is not None else rhs.sres
        return Spectrum(self.wave, flux, error=error, resolution=sres, log=self.log)

    def _add_spectrum(self, rhs):
        """
        rhs must have type Spectrum
        """
        if not numpy.array_equal(rhs.wave, self.wave):
            raise NotImplementedError('To perform arithmetic on spectra, their wavelength arrays '
                                      'must be identical.')
        flux = self.flux + rhs.flux
        if self.error is None and rhs.error is None:
            error = None
        else:
            error = numpy.zeros(self.size, dtype=float)
            if self.error is not None:
                error += numpy.square(self.error)
            if rhs.error is not None:
                error += numpy.square(rhs.error)
            error = numpy.sqrt(error)
        if self.sres is not None and rhs.sres is not None \
                and not numpy.array_equal(self.sres, rhs.sres):
            warnings.warn('Spectral resolution is not correctly propagated.')
        sres = self.sres if self.sres is not None else rhs.sres
        return Spectrum(self.wave, flux, error=error, resolution=sres, log=self.log)

    def __len__(self):
        return self.interpolator.x.size

    def __getitem__(self, s):
        """
        Access the flux data directly via slicing.
        """
        return self.interpolator.y[s]

    def _frequency(self):
        """Calculate the frequency in terahertz."""
        return 10*astropy.constants.c.to('km/s').value/self.wave

    def interp(self, w):
        """
        Linearly interpolate the flux at a provided wavelength.

        Args:
            w (:obj:`float`, `numpy.ndarray`_):
                Wavelength in angstroms. Can be a single wavelength
                or a wavelength array.
        """
        if not isinstance(w, numpy.ndarray) and w > self.interpolator.x[0] \
                and w < self.interpolator.x[-1]:
            return self.interpolator(w)

        indx = (w > self.interpolator.x[0]) & (w < self.interpolator.x[-1])
        sampled = numpy.zeros_like(w, dtype=float)
        sampled[indx] = self.interpolator(w[indx])
        return sampled

    @classmethod
    def from_file(cls, fitsfile, waveext='WAVE', fluxext='FLUX', resext=None):
        """
        Construct the spectrum using a fits file.
        """
        hdu = fits.open(fitsfile)
        wave = hdu[waveext].data
        flux = hdu[fluxext].data
        sres = None if resext is None else hdu[resext].data
        return cls(wave, flux, resolution=sres)

    def wavelength_step(self):
        """
        Return the wavelength step per pixel.

        TODO: FIX THIS!!  It shouldn't use spectral_coordinate_step to get the mean dw.
        """
        return angstroms_per_pixel(self.wave, log=self.log, regular=True)
#        # TODO: Lazy load and then keep this?
#        dw = spectral_coordinate_step(self.wave, log=self.log)
#        if self.log:
#            dw *= numpy.log(10.)*self.wave
#        return dw

    def frequency_step(self):
        """
        Return the frequency step per pixel in THz.
        """
        return 10*astropy.constants.c.to('km/s').value*self.wavelength_step()/self.wave/self.wave

    def magnitude(self, wavelength=None, band=None, system='AB'):
        """
        Calculate the magnitude of the object in the specified band.

        If no arguments are provided, the magnitude is calculated for
        the entire :attr:`flux` vector. Otherwise, the magnitude is
        calculated at a single wavelength (see ``wavelength``) or
        over a filter (see ``band``).

        Args:
            wavelength (:obj:`float`, optional):
                The wavelength at which to calculate the magnitude.
            band (:class:`~enyo.etc.efficiency.FilterResponse`, optional):
                Object with the filter response function
            system (:obj:`str`, optional):
                Photometric system.  Currently must be ``AB``.

        Returns:
            :obj:`float`, `numpy.ndarray`_: The one or more magnitude
            measurements, depending on the input.

        Raises:
            NotImplementedError:
                Raised if the photometric system is not known.
        """
        if wavelength is not None and band is not None:
            warnings.warn('Provided both wavelength and band; wavelength takes precedence.')
        # TODO: Check input.
        if system == 'AB':
            if self.nu is None:
                self._frequency()
            if self.fnu is None:
                # Flux in microJanskys
                self.fnu = convert_flux_density(self.wave, self.flux)
            if wavelength is not None:
                fnu = self.fnu[numpy.argmin(numpy.absolute(self.wave - wavelength))]
            elif band is not None:
                dnu = self.frequency_step()
                fnu = numpy.sum(band(self.wave)*self.fnu*dnu) / numpy.sum(band(self.wave)*dnu)
            else:
                fnu = self.fnu
            return -2.5*numpy.log10(fnu*1e-29) - 48.6

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

    def rescale_magnitude(self, new_mag, wavelength=None, band=None, system='AB'):
        """
        Rescale the spectrum to an input magnitude.

        Must provide either ``wavelength`` or ``band``. The object is
        edited in-place.

        Args:
            new_mag (:obj:`float`):
                The target magnitude
            wavelength (:obj:`float`, optional):
                The wavelength at which to calculate the magnitude.
            band (:class:`~enyo.etc.efficiency.FilterResponse`, optional):
                Object with the filter response function
            system (:obj:`str`, optional):
                Photometric system.  Currently must be ``AB``.

        Raises:
            ValueError:
                Raised if neither ``wavelength`` nor ``band`` are provided.
            NotImplementedError:
                Raised if the photometric system is not known.
        """
        if wavelength is None and band is None:
            raise ValueError('Must provide either wavelength or the bandpass filter.')
        dmag = new_mag - self.magnitude(wavelength=wavelength, band=band, system=system)
        if system == 'AB':
            return self.rescale(numpy.power(10., -dmag/2.5))
        raise NotImplementedError('Photometric system {0} not implemented.'.format(system))

    def photon_flux(self, inplace=True):
        r"""
        Convert the spectrum from 1e-17 erg/s/cm^2/angstrom to
        photons/s/cm^2/angstrom.

        If `inplace is True`, the spectrum is modified in place and
        None is returned; otherwise the converted flux vector is
        returned.
        """
        ergs_per_photon = astropy.constants.h.to('erg s') * astropy.constants.c.to('angstrom/s') \
                            / (self.wave * astropy.units.angstrom)
        return self.rescale(1e-17 / ergs_per_photon.value) if inplace \
                    else self.interpolator.y * 1e-17 / ergs_per_photon.value

    def plot(self, ax=None, show=False, **kwargs):
        _ax = pyplot.subplot() if ax is None else ax
        _ax.plot(self.wave, self.flux, **kwargs)
        if show:
            pyplot.show()
        return _ax

    def resample(self, wave, log=False):
        """
        Resample the spectrum to a new wavelength array.

        Args:
            wave (`numpy.ndarray`_):
                New wavelength array. Must be linearly or
                log-linearly sampled.
            log (:obj:`bool`, optional):
                Flag that the wavelength array is log-linearly
                sampled.

        Returns:
            :class:`Spectrum`: Returns a a resampled version of
            itself. TODO: The resampled version currently looses any
            error or resolution vectors...
        """
        # TODO: Make this better!
        rng = wave[[0,-1]]
        if log:
            rng = numpy.log10(rng)
        r = Resample(self.flux, x=self.wave, inLog=self.log, newRange=rng, newpix=wave.size,
                     newLog=log)
        return Spectrum(r.outx, r.outy, log=log)

    def redshift(self, z):
        """
        Redshift the spectrum.

        Spectrum is in 1e-17 erg/s/cm^2/angstrom, so this shifts the
        wavelength vector by 1+z and rescales the flux by 1+z to keep
        the flux per *observed* wavelength.  S/N is kept fixed.
        """
        self.interpolator.x *= (1+z)
        self.rescale(1/(1+z))


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
        units (:obj:`str`, array-like, optional):
            The units of the provided FWHM data. Must be either
            'km/s' or 'ang'. Can be a single string or an array with
            the units for each provided value.
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
    def __init__(self, wave, flux, restwave, fwhm, units='ang', redshift=0.0, continuum=None,
                 resolution=None, log=False):

        _flux = numpy.atleast_1d(flux).ravel()
        nlines = _flux.size
        _restwave = numpy.atleast_1d(restwave).ravel()
        if _restwave.size != nlines:
            raise ValueError('Number of rest wavelengths does not match the number of fluxes.')
        _fwhm = numpy.atleast_1d(fwhm).ravel()
        if _fwhm.size != nlines:
            raise ValueError('Number of FWHM values does not match the number of fluxes.')
        _units = numpy.atleast_1d(units).ravel()
        # Check the input
        if not numpy.all(numpy.isin(_units, ['km/s', 'ang'])):
            raise ValueError('FWHM units must be \'km/s\' or \'ang\'.')
        if _units.size == 1:
            _units = numpy.repeat(_units, _flux.size)
        if _units.size != nlines:
            raise ValueError('Number of unit values does not match the number of fluxes.')

        if resolution is not None and hasattr(resolution, '__len__') \
                and len(wave) != len(resolution):
            raise ValueError('Resolution vector must match length of wavelength vector.')
        if continuum is not None and len(wave) != len(continuum):
            raise ValueError('Continuum vector must match length of wavelength vector.')

        # Set the line parameters
        _linewave = _restwave*(1+redshift)
        indx = (_linewave > wave[0]) & (_linewave < wave[-1])
        if not numpy.any(indx):
            raise ValueError('Redshifted lines are all outside of the provided wavelength range.')

        sig2fwhm = numpy.sqrt(8.0 * numpy.log(2.0))
        sigma = _fwhm/sig2fwhm
        # Convert the FWHM as needed based on the sampling and units
        in_ang = _units == 'ang'
        in_kms = _units == 'km/s'
        if log and numpy.any(in_ang):
            # Convert to km/s
            sigma[in_ang] = astropy.constants.c.to('km/s').value*sigma[in_ang]/_linewave[in_ang]
        elif not log and numpy.any(in_kms):
            # Convert to angstroms
            sigma[in_kms] = _linewave[in_kms]*sigma[in_kms]/astropy.constants.c.to('km/s').value

        # Add the instrumental resolution in quadrature to the
        # intrinsic width, if the resolution is provided
        if resolution is None:
            _resolution = None
        else:
            _resolution = resolution if hasattr(resolution, '__len__') \
                                else numpy.full_like(wave, resolution, dtype=float)
            sigma_inst = astropy.constants.c.to('km/s').value/_resolution/sig2fwhm if log else \
                            wave/_resolution/sig2fwhm
            interp = interpolate.interp1d(wave, sigma_inst, assume_sorted=True, bounds_error=False,
                                          fill_value=0.)
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
        spectrum = numpy.zeros(wave.size, dtype=float) if continuum is None else continuum.copy()
        profile = IntegratedGaussianLSF()
        for i in range(nlines):
            if not indx[i]:
                continue
            p = profile.parameters_from_moments(_flux[i], _linepix[i], sigma[i])
            spectrum += profile(pix, p)

        # Instantiate
        super(EmissionLineSpectrum, self).__init__(wave, spectrum, resolution=_resolution, log=log)


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
    def __init__(self, wave, resolution=None, log=False):
        norm = numpy.power(10., 29 - 48.6/2.5)  # Reference flux in microJanskys
        fnu = numpy.full_like(wave, norm, dtype=float)
        flambda = convert_flux_density(wave, fnu, density='Hz')
        super(ABReferenceSpectrum, self).__init__(wave, flambda, resolution=resolution, log=log)

