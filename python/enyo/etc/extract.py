#!/bin/env/python3
# -*- encoding utf-8 -*-
import numpy
from scipy import special

# TODO: Include an optimal extraction class or method

class Extraction:
    """
    fwhm is the number of detector pixels per fwhm
    
    width is the number of pixels for the extraction.

    FWHM can be fractional pixels; for now width must be whole pixels.

    """
    def __init__(self, detector, spatial_fwhm=3, spatial_width=3,
                 spectral_fwhm=3, spectral_width=3):
        self.detector = detector

        # All in pixel units
        self.spatial_fwhm = spatial_fwhm
        self.spatial_width = int(spatial_width)
        self.spectral_fwhm = spectral_fwhm
        self.spectral_width = int(spectral_width)

        self._get_profile()

    @staticmethod
    def _pixelated_gaussian(fwhm, width):
        # Pixel edges in units of the profile dispersion
        edges = width/fwhm * numpy.sqrt(8*numpy.log(2)) * numpy.linspace(-1, 1, width+1)/2
        profile = (special.erf(edges[1:]/numpy.sqrt(2))
                        - special.erf(edges[:-1]/numpy.sqrt(2)))/2.
        return edges, profile

    def _get_profile(self):
        """
        Construct the spatial profile over the extraction aperture.
        """
        self.spatial_edges, self.spatial_profile \
                = Extraction._pixelated_gaussian(self.spatial_fwhm, self.spatial_width)
        self.spectral_edges, self.spectral_profile \
                = Extraction._pixelated_gaussian(self.spectral_fwhm, self.spectral_width)

        # 2D profile? ...
#        self.profile = self.spectral_profile[:,None]*self.spatial_profile[None,:]

    def sum_signal_and_noise(self, object_flux, sky_flux, exposure_time, spectral_pixels=3.):
        """
        Get the signal and noise from a summed extraction. NO SUMMING
        IS DONE SPECTRALLY.

        Fluxes should be in electrons per second per resolution
        element or per angstrom. Flux per second isn't explicitly
        necessary for the calculation, but it allows for a
        consistency with the dark current calculation.

        object_flux and sky_flux can be spectra or individual values

        Returned fluxes and errors are in electrons per resolution
        element or per angstrom.

        The input spectra are not resampled. The signal and variance
        are returned in such that the S/N is per resolution element.
        The `pixels_per_resolution` is used to set the how many
        pixels to use when accounting for read noise.

        """
        # Get the variance in each pixel
        extract_box_n = len(self.spatial_profile) * spectral_pixels
        read_var = numpy.square(self.detector.rn) * extract_box_n
        if isinstance(object_flux, numpy.ndarray):
            ext_obj_flux = numpy.sum(object_flux[None,:]*self.spatial_profile[:,None], axis=0) \
                                * exposure_time
            ext_sky_flux = numpy.sum(sky_flux[None,:]*self.spatial_profile[:,None], axis=0) \
                                * exposure_time
            read_var = numpy.full_like(ext_obj_flux, read_var, dtype=float)
        else:
            ext_obj_flux = numpy.sum(object_flux*self.spatial_profile) * exposure_time
            ext_sky_flux = numpy.sum(sky_flux[None,:]*self.spatial_profile[:,None], axis=0) \
                                * exposure_time

        obj_shot_var = ext_obj_flux + self.detector.dark * extract_box_n * exposure_time
        sky_shot_var = ext_sky_flux + self.detector.dark * extract_box_n * exposure_time

        return ext_obj_flux, obj_shot_var, ext_sky_flux, sky_shot_var, read_var

    # TODO: Add optimal_signal_and_noise()

#    def sum_noise_budget(self, object_flux, sky_flux):
#        """
#        Get the noise budget from a summed extraction.
#
#        Fluxes should be in electrons per resolution element
#
#        object_flux and sky_flux can be spectra
#        """
#        # Get the variance in each pixel
#        n_pixels = numpy.nprod(numself.profile.shape)
#        if isinstance(object_flux, numpy.ndarray):
#            object_err = numpy.sqrt(numpy.sum(object_flux[None,None,:] * self.profile[:,:,None],
#                                              axis=0))
#            sky_err = numpy.sqrt(numpy.sum(sky_flux[None,None,:] * self.profile[:,:,None], axis=0))
#
#            dark_err = numpy.full_like(object_err, numpy.sqrt(dark*n_pixels))
#            read_err = numpy.full_like(object_err, detector.rn*numpy.sqrt(n_pixels))
#            return object_err, sky_err, dark_err, read_err
#
#        return numpy.sqrt(object_flux*self.efficiency), numpy.sqrt(sky_flux*self.efficiency), \
#                numpy.sqrt(dark*n_pixels), detector.rn*numpy.sqrt(n_pixels)
    
    @property
    def spatial_efficiency(self):
        return numpy.sum(self.spatial_profile)


