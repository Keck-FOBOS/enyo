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
        self.profile = (special.erf(edges[1:]/numpy.sqrt(2)) 
                                - special.erf(edges[:-1]/numpy.sqrt(2)))/2.
        self.n_spectral = spectral_fwhm * spectral_width

    def sum_snr(self, object_flux, sky_flux, rn, dark=0.):
        """
        Get the S/N from a summed extraction.

        Fluxes should be in electrons to match units from detector
        """
        # Get the variance in each pixel
        variance = (object_flux + sky_flux)*self.profile + dark + numpy.square(rn)
        variance *= self.n_spectral
        # Just perform a summed extraction
        return numpy.sum(self.profile)*self.n_spectral*object_flux/numpy.sqrt(numpy.sum(variance))

