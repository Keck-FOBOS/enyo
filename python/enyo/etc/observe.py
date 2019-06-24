#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Class to construct an observation

Extraction:
    - fiber-extraction aperture (pixels, optimal?)
    - fiber PSF FWHM on detector (pixels)

    - spectral resolution (R)
    - dispersion on detector (A per pixel)

Detector:
    - arcsec / pixel (spatial)
    - angstrom / pixel (spectral)
    - detector readnoise (e-)
    - detector darkcurrent (e-/hour)

Throughput
    - spectrograph throughput
        - detector QE
        - camera efficiency
        - grating efficiency
        - focal-plane to grating tansmission efficiency
            - foreoptics, lenslet, coupling, fiber transmission, FRD,
              collimator, dichroic(s)

    - top-of-telescope throughput
        - telescope efficiency
        - spectrograph efficiency

Point-source Aperture losses
    - fiber diameter (arcsec, mm)
    - focal-plane plate scale (arcsec/mm)
    - seeing (arcsec)
    - pointing error (arcsec)

*Source:
    - source spectrum (1e-17 erg/s/cm^2/angstrom)

    - point source
        - source mag (at wavelength/in band above)

    - extended source
        - source surface brightness (at wavelength/in band above)
        - source size (kpc/arcsec)
        - cosmology

    - source velocity/redshift

*Sky:
    - sky spectrum (1e-17 erg/s/cm^2/angstrom)

Observation:
    - obs date
    - lunar phase/illumination (days, fraction)
    - sky coordinates
    - wavelength for calculation (angstroms)
    - band for calculation
    - sky surface brightness (mag/arcsec^2, at wavelength/in band above)
    - airmass
    - exposure time (s)

Telescope:
    - location
    - telescope diameter (or effective) (m^2)
    - central obstruction

"""
import os
import warnings
import numpy

from scipy import signal

from matplotlib import pyplot

from . import source, efficiency, telescopes, spectrum, extract, aperture, util

class Observation:
    """
    Observation of the sky with or without a non-terrestrial source.

    Args:
        magnitude (scalar-like):
            Apparent magnitude of the source.
        exposure_time (scalar-like):
            The total exposure time in seconds.
        seeing (scalar-like):
            The FWHM of the Gaussian seeing distribution in arcsec.
        airmass (scalar-like):
            The airmass of the observation.
        resolution (array-like):
            1D spectral resolution (:math:`R = \lambda/\Delta\lambda`).
            Must be a scalar or match the length of the source spectrum
            wavelength vector.
        rn (scalar-like):
            Detector read noise in electrons.
        dark (scalar-like):
            Detector dark current in electrons per second.
        qe (scalar-like):
            Detector quantum efficiency (wavelength indepedent).
        spectrograph_throughput
            (:class:`efficiency.SpectrographThroughput`, optional):
            The spectrograph throughput objects.  If not provided, uses
            data in data/efficiency/fiber_wfos_throughput.db.
        fiber_diameter (scalar-like, optional):
            On-sky diameter of the fiber in arcseconds.
        spectral_fwhm (scalar-like, optional):
            The FHWM of the Gaussian line-spread function of the
            instrument at the detector in pixels.
        spectral_width (scalar-like, optional):
            The extraction width of the spectral pixel in number of FWHM.
        spatial_fwhm (scalar-like, optional):
            The FHWM of the Gaussian point-spread function of the
            fiber on the detector in pixels.
        spatial_width (scalar-like, optional):
            The extraction width of the spatial pixel in number of FWHM.
        surface_brightness (scalar-like, optional):
            Central surface brightness of the object is AB mag /
            arcsec^2.
        reff (scalar-like, optional):
            The effective (half-light) radius in pixels.
        sersic_index (scalar-like, optional):
            The Sersic index.
        redshift (scalar-like, optional):
            Redshift of the source.
        spectrum (:obj:`str`, :class:`spectrum.Spectrum`, optional):
            The spectrum of the source.  Can be a spectrum object, a
            string used to set the object, or a file name read using
            :func:`enyo.etc.spectrum.Spectrum.from_file`.  Cannot be
            None.
        sky (:obj:`str`, :class:`spectrum.Spectrum`, optional):
            The spectrum of the night-sky.  Can be a spectrum object, a
            string used to set the object, or a file name read using
            :func:`enyo.etc.spectrum.Spectrum.from_file`.  Cannot be
            None.
        normalizing_band (:obj:`str`, optional):
            Rest-frame broad-band filter in which the magnitude or
            surface brightness is defined.


    if atmospheric throughput is not provided, assume Maunakea curve
    if airmass is None, use default defined by atmospheric throughput class/instance

    if sky spectrum is None, assume dark Maunakea sky
    if sky distribution is None, assume constant

    if source or source distribution are None (must provide both), assume a sky-only exposure

    if extraction is None, provide 2d result; otherwise return 1D extraction

    """
    def __init__(self, spec_aperture, spectrograph, exposure_time, atmospheric_throughput=None,
                 airmass=None, onsky_source_distribution=None, source_spectrum=None,
                 sky_spectrum=None, sky_distribution=None, extraction=None): 

#                  system_throughput,
#                 detector, exposure_time, extraction):

#        # Rescale source spectrum to the input magnitude or surface
#        # brightness
#        print('band')
#        self.band = efficiency.FilterResponse(band=self.normalizing_band)
#        print('source rescale')
#        self.source_spectrum.rescale_magnitude(self.band, self.magnitude 
#                                if self.surface_brightness is None else self.surface_brightness)

        # Save the input or use defaults
        self.source = onsky_source_distribution

        # In the current implementation, the source spectrum is expected
        # to be:
        #   - independent of position within the source
        #   - the source flux density integrated over the full source
        #     distribution; i.e., units are, e.g., erg/s/cm^2/angstrom
        self.wave = source_spectrum.wave.copy()
        self.source_spectrum = source_spectrum

        # Sky spectrum is expected to be independent of position within
        # the aperture and be the sky flux density per unit area, where
        # the unit area is defined by the aperture object (arcsec^2)
        # i.e., the units are, e.g., erg/s/cm^2/angstrom/arcsec^2
        self.sky_spectrum = spectrum.Spectrum(self.wave, sky_spectrum.interp(self.wave))

        # Check that sky spectrum and source spectrum have the same
        # length!

        self.atmospheric_throughput = atmospheric_throughput
        self.telescope = telescope
        # TODO: Allow for a list of apertures or a single aperture and a
        # list of offsets.
        self.aperture = focal_plane_aperture

        self.system_throughput = system_throughput
        self.detector = detector
        self.exptime = exposure_time
        self.extraction = extraction

        # Get the "aperture efficiency"
        self.aperture_efficiency = self.aperture.integrate_over_source(self.source) \
                                        / self.source.integral

        # Get the total object flux incident on the focal plane in
        # electrons per second per angstrom
        _object_flux = self.source_spectrum.photon_flux() \
                                * self.telescope.area \
                                * self.atmospheric_throughput(self.wave) \
                                * self.system_throughput(self.wave) \
                                * self.aperture_efficiency \
                                * self.detector.efficiency(self.wave)

        # Total sky flux in electrons per second per angstrom; the
        # provided sky spectrum is always assumed to be uniform over the
        # aperture
        _sky_flux = self.sky_spectrum.photon_flux() * self.aperture.area \
                                * self.telescope.area \
                                * self.system_throughput(self.wave) \
                                * self.detector.efficiency(self.wave)

        # Observe and extract the source
        self.total_flux, self.sky_flux, self.shot_variance, self.read_variance \
                = self.extraction.sum_signal_and_noise(_object_flux, _sky_flux, self.exptime,
                                                       wave=self.wave)

    def simulate(self):
        """
        Return a simulated spectrum
        """
        # Draw from a Poisson distribution for the shot noise
        shot_draw = numpy.random.poisson(lam=self.shot_variance)
        # Draw from a Gaussian distribution for the read noise
        read_draw = numpy.random.normal(scale=numpy.sqrt(self.read_variance))
        return spectrum.Spectrum(self.wave,
                                 self.total_flux - self.sky_flux + shot_draw + read_draw,
                                 error=numpy.sqrt(self.shot_variance + self.read_variance),
                                 log=self.detector.log)

    def snr(self):
        return spectrum.Spectrum(self.wave,
                                 (self.total_flux - self.sky_flux)
                                        / numpy.sqrt(self.shot_variance + self.read_variance))


def monochromatic_image(sky, spec_aperture, spec_kernel, platescale, pixelsize, onsky_source=None,
                        scramble=False):
    """
    Generate a monochromatic image of the sky, with or with out a
    non-terrestrial source, taken by a spectrograph through an
    aperture.

    .. warning::
        - May resample the `source` and `spec_kernel` maps.

    .. todo::
        - Add effect of differential atmospheric refraction
        - Allow a force map size and pixel sampling

    Args:
        sky (:class:`enyo.etc.source.OnSkySource`):
            Sky flux distribution.
        spec_aperture (:class:`enyo.etc.aperture.Aperture`):
            Spectrograph aperture. Aperture is expected to be
            oriented with the dispersion along the first axis (e.g.,
            the slit width is along the abcissa).
        spec_kernel (:class:`enyo.etc.kernel.SpectrographGaussianKernel`):
            Convolution kernel describing the point-spread function
            of the spectrograph.
        platescale (:obj:`float`):
            Platescale in mm/arcsec at the detector
        pixelsize (:obj:`float`):
            Size of the detector pixels in mm.
        onsky_source (:class:`enyo.etc.source.OnSkySource`, optional):
            On-sky distribution of the source flux. If None, only sky
            is observed through the aperture.
        scramble (:obj:`bool`, optional):
            Fully scramble the source light passing through the
            aperture. This should be False for slit observations. For
            fiber observations, this should be True and makes the
            nominal assumption that the focal plane incident on the
            fiber face is perfectly scrambled.
    """
    # Check input
    if onsky_source is not None:
        if onsky_source.sampling is None or onsky_source.size is None:
            warnings.warn('Source was not provided with an initial map sampling; doing so now '
                          'with default sampling and size.')
            onsky_source.make_map()

    # Detector pixel scale
    pixelscale = pixelsize/platescale

    # Assume the sampling of the source is provided with the maximum
    # allowed pixel size. Determine a pixel size that is no more than
    # this, up to an integer number of detector pixels. Use an integer
    # number specifically so that the result of the kernel convolution
    # can be simply rebinned to match the detector pixel size.
    # `sampling` is in arcsec per pixel
    sampling = pixelscale if onsky_source is None else min(onsky_source.sampling, pixelscale)
    oversample = int(pixelscale/sampling)+1 if sampling < pixelscale else 1
    sampling /= oversample

    # Assume the size of the image properly samples the source.
    # Determine a map size that at least encompasses the input source
    # and the input aperture. The factor of 1.5 is ad hoc; could likely
    # be lower. `size` is in arcsecA
    dx, dy = 1.5*numpy.diff(numpy.asarray(spec_aperture.bounds).reshape(2,-1), axis=0).ravel()
    size = max(dx,dy) if onsky_source is None else max(onsky_source.size, dx, dy)

    # TODO: Below alters `source` and `spec_kernel`. Should maybe
    # instead save the old sampling and size and then resample back to
    # the input before returning.

    # Resample the distribution maps. Classes OnSkySource and Aperture
    # use arcsecond units.
    if onsky_source is not None:
        onsky_source.make_map(sampling=sampling, size=size)
    sky.make_map(sampling=sampling, size=size)
    ap_img = spec_aperture.response(sky.x, sky.y)

    # However, SpectrographGaussianKernel uses mm, so we need to
    # convert sampling from arcsec/pixel to mm/pixel using the
    # platescale.
    spec_kernel.resample(pixelscale=platescale*sampling)

    # Construct the image. Note that scrambling simply scales the
    # aperture image by the flux that enters it; otherwise, the source
    # image is just attenuated by the aperture response map.
    source_img = sky.data if onsky_source is None else onsky_source.data + sky.data
    input_img = ap_img*numpy.sum(source_img*ap_img)/numpy.sum(ap_img) if scramble \
                        else source_img*ap_img

    # Convolve it with the spectrograph imaging kernel
    mono_img = signal.fftconvolve(input_img, spec_kernel.array, mode='same')
    # Return the image, downsampling if necessary
    return util.boxcar_average(mono_img, oversample) if oversample > 1 else mono_img

