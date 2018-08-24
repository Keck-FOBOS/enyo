#!/bin/env/python3
# -*- encoding utf-8 -*-
import numpy
from scipy import special

class Extraction:
    """
    spatial_fwhm is the number of detector pixels per fwhm
    
    width is the number of fwhm to extract
    """
    def __init__(self, spatial_fwhm=3, spatial_width=1, spectral_fwhm=3, spectral_width=1):
        self.edges = spatial_width * numpy.sqrt(8*numpy.log(2)) \
                        * numpy.linspace(-1, 1, spatial_fwhm+1)/2
        self.profile = (special.erf(self.edges[1:]/numpy.sqrt(2)) 
                                - special.erf(self.edges[:-1]/numpy.sqrt(2)))/2.
        n_spectral = int(spectral_fwhm * spectral_width)
        self.profile = numpy.repeat(self.profile.reshape(1,-1)/n_spectral, n_spectral, axis=0)

    def sum_snr(self, object_flux, sky_flux, rn, dark=0.):
        """
        Get the S/N from a summed extraction.

        Fluxes should be in electrons per resolution element

        object_flux and sky_flux can be spectra
        """
        if isinstance(object_flux, numpy.ndarray):
            # Get the variance in each pixel
            variance = (object_flux + sky_flux)[None,None,:]*self.profile[:,:,None] + dark \
                            + numpy.square(rn)
            return object_flux / numpy.sqrt(numpy.sum(variance.reshape(-1,object_flux.size),
                                                      axis=0))
        # Get the variance in each pixel
        variance = (object_flux + sky_flux)*self.profile + dark + numpy.square(rn)
        return object_flux/numpy.sqrt(numpy.sum(variance))

    def sum_noise_budget(self, object_flux, sky_flux, rn, dark=0.):
        """
        Get the noise budget from a summed extraction.

        Fluxes should be in electrons per resolution element

        object_flux and sky_flux can be spectra
        """
        # Get the variance in each pixel
        n_pixels = numpy.nprod(numself.profile.shape)
        if isinstance(object_flux, numpy.ndarray):
            object_err = numpy.sqrt(numpy.sum(object_flux[None,None,:] * self.profile[:,:,None],
                                              axis=0))
            sky_err = numpy.sqrt(numpy.sum(sky_flux[None,None,:] * self.profile[:,:,None], axis=0))

            dark_err = numpy.full_like(object_err, numpy.sqrt(dark*n_pixels))
            read_err = numpy.full_like(object_err, rn*numpy.sqrt(n_pixels))
            return object_err, sky_err, dark_err, read_err

        return numpy.sqrt(object_flux*self.efficiency), numpy.sqrt(sky_flux*self.efficiency), \
                numpy.sqrt(dark*n_pixels), rn*numpy.sqrt(n_pixels)
    
    @property
    def efficiency(self):
        return numpy.sum(self.profile)


