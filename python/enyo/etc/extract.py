from IPython import embed

import numpy
from scipy import special

# TODO: Include an optimal extraction class or method

class Extraction:
    """
    fwhm is the number of detector pixels per fwhm
    
    width is the number of pixels for the extraction.

    FWHM can be fractional pixels; for now width must be whole pixels.

    Profile describes how the light is distributed across the
    detector pixels. For now, use ``profile='gaussian'`` for a fiber
    aperture, and ``profile='uniform'`` for a slit aperture, but that
    should be improved. For a uniform profile, the FWHM and width
    should be the same, but the FWHM is never used.

    Spectral profile is not used.

    .. todo::

        Allow the 2D profile to be provided directly.

    """
    def __init__(self, detector, spatial_fwhm=3, spatial_width=3,
                 spectral_fwhm=3, spectral_width=3, profile='gaussian'):
        self.detector = detector

        # All in pixel units
        self.spatial_fwhm = spatial_fwhm
        self.spatial_width = int(spatial_width)
        self.spectral_fwhm = spectral_fwhm
        self.spectral_width = int(spectral_width)

        self._get_profile(profile)

    @staticmethod
    def _pixelated_gaussian(fwhm, width):
        # Pixel edges in units of the profile dispersion
        edges = width/fwhm * numpy.sqrt(8*numpy.log(2)) * numpy.linspace(-1, 1, width+1)/2
        profile = (special.erf(edges[1:]/numpy.sqrt(2))
                        - special.erf(edges[:-1]/numpy.sqrt(2)))/2.
        return edges, profile

    @staticmethod
    def _pixelated_uniform(width):
        # Pixel edges in units of the profile dispersion
        edges = width * numpy.linspace(-1, 1, width+1)/2
        profile = numpy.full(width, 1/width, dtype=float)
        return edges, profile

    def _get_profile(self, profile):
        """
        Construct the spatial profile over the extraction aperture.
        """
        if profile == 'gaussian':
            self.spatial_edges, self.spatial_profile \
                    = Extraction._pixelated_gaussian(self.spatial_fwhm, self.spatial_width)
            self.spectral_edges, self.spectral_profile \
                    = Extraction._pixelated_gaussian(self.spectral_fwhm, self.spectral_width)
            return

        if profile == 'uniform':
            self.spatial_fwhm = self.spatial_width
            self.spatial_edges, self.spatial_profile \
                    = Extraction._pixelated_uniform(self.spatial_width)
            self.spectral_fwhm = self.spectral_width
            self.spectral_edges, self.spectral_profile \
                    = Extraction._pixelated_uniform(self.spectral_width)
            return

        raise ValueError('{0} profile not recognized.  Must be \'gaussian\' or \'uniform\'')

        # 2D profile? ...
#        self.profile = self.spectral_profile[:,None]*self.spatial_profile[None,:]

    def sum_signal_and_noise(self, object_flux, sky_flux, exposure_time, spectral_width=1.):
        """
        Get the signal and noise from a summed extraction.

        Fluxes should be in electrons per second per resolution
        element, per angstrom, or per pixel.

        The primary operation is to assume the input spectrum is
        distributed in a spatial and spectral profile that is summed.
        This operation *only* sums over the spatial profile. The
        spectral width provided is used to set the number of pixels
        extracted to construct the desired S/N units (per resolution
        element, angstrom, pixel). To get the S/N per pixel, use the
        default. Otherwise, ``spectral_width`` is, e.g., the number
        of pixels per angstrom.

        object_flux and sky_flux can be spectra or individual values,
        but they must match. I.e., if object_flux is a vector,
        sky_flux should be also.

        Returned fluxes and errors are in electrons per resolution
        element, per angstrom, or per pixel, depending on the input.

        """
        # Get the variance in each pixel
        extract_box_n = len(self.spatial_profile) * spectral_width
        read_var = numpy.square(self.detector.rn) * extract_box_n
        if isinstance(object_flux, numpy.ndarray):
            ext_obj_flux = numpy.sum(object_flux[None,:]*self.spatial_profile[:,None], axis=0)
            ext_sky_flux = numpy.sum(sky_flux[None,:]*self.spatial_profile[:,None], axis=0)
            if not isinstance(spectral_width, numpy.ndarray):
                read_var = numpy.full_like(ext_obj_flux, read_var, dtype=float)
        else:
            if isinstance(spectral_width, numpy.ndarray):
                raise TypeError('Cannot use vector of spectral widths with individual fluxes.')
            ext_obj_flux = numpy.sum(object_flux*self.spatial_profile)
            ext_sky_flux = numpy.sum(sky_flux*self.spatial_profile)

        ext_obj_flux *= exposure_time
        ext_sky_flux *= exposure_time
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


