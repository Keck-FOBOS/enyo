#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Define an on-sky source distribution
"""
import numpy

from scipy import signal, special

from astropy.modeling import functional_models

# TODO: Need to learn how to use abstract classes!  Inherit from
# numpy.ndarray?
class Source:
    def __init__(self):
        self.x = None
        self.y = None
        self.data = None
    
    def minimum_sampling(self):
        pass

    def minimum_size(self):
        pass

    def make_map(self, sampling=None, size=None):
        """Generate a map of the distribution."""
        # Get the map x and y coordinates
        _size = self.minimum_size() if size is None else size
        _samp = self.minimum_sampling() if sampling is None else sampling
        pixsize = numpy.ceil(_size/_samp).astype(int)
        if pixsize % 2 == 0:
            pixsize += 1
        c = (pixsize-1)*numpy.linspace(-0.5,0.5,pixsize)*_samp
        self.x, self.y = numpy.meshgrid(c[::-1],c)
        # Sample it
        self.data = self.__call__(self.x, self.y)

    def __getitem__(self, s):
        """Slice the map."""
        if self.data is None:
            raise ValueError('Distribution data is not defined!')
        return self.data[s]

    @property
    def shape(self):
        return () if self.data is None else self.data.shape


class OnSkyGaussian(functional_models.Gaussian2D, Source):
    """
    An on-sky Gaussian distribution.

    Args:
        fwhm (scalar-like):
            The FWHM of the Gaussian in *arcseconds*.
        center (scalar-like, optional):
            The coordinates of the Gaussian center in arcseconds.
        ellipticity (scalar-like, optional):
            The ellipticity (1-b/a) of an elliptical Gaussian
            distribution.
        position_angle (scalar-like, optional):
            The position angle for the elliptical Gaussian distribution,
            defined as the angle from N through E.  The coordinate
            system is defined with positive offsets (in RA) toward the
            east, meaning lower pixel indices.
        sampling (scalar-like, optional):
            Sampling of a generated map in arcseconds per pixel.
            Default is set by :func:`minimum_sampling`.
        size (scalar-like, optional):
            Size of the image to generate of the distribution in
            *arceconds* along one of the axes.  The map is square.
            Default is defined by :func:`minimum_size`.
    """
    def __init__(self, fwhm, center=None, ellipticity=None, position_angle=None, sampling=None,
                 size=None):

        # Define internals
        self.fwhm = float(fwhm)
        self.ellipticity = 0 if ellipticity is None else ellipticity
        self.position_angle = 0 if position_angle is None else position_angle
        sig2fwhm = numpy.sqrt(8*numpy.log(2))
        major_sigma = self.fwhm/sig2fwhm
        minor_sigma = major_sigma * (1-self.ellipticity)

        # Instantiate the functional_models.Gaussian2D object
        super(OnSkyGaussian, self).__init__(amplitude=1/(2*major_sigma*minor_sigma*numpy.pi),
                                            x_mean=0 if center is None else center[0],
                                            y_mean=0 if center is None else center[1],
                                            x_stddev=minor_sigma, y_stddev=major_sigma,
                                            theta=-numpy.radians(self.position_angle))

        # Set the integral to be normalized
        self.integral = 1.0

        # Set the map if requested
        if size is not None:
            self.make_map(sampling=sampling, size=size)

    def get_integral(self):
        sig2fwhm = numpy.sqrt(8*numpy.log(2))
        major_sigma = self.fwhm/sig2fwhm
        minor_sigma = major_sigma * (1-self.ellipticity)
        return 2*numpy.pi*major_sigma*minor_sigma

    def minimum_sampling(self):
        r"""
        Return the minimum sampling in arcseconds per pixels.  Currently
        :math:`{\rm FWHM}/2`.
        """
        return self.fwhm/2.

    def minimum_size(self):
        r"""
        The minimum size that should be used for the distribution map in
        arcseconds.  Currently :math:`2\ {\rm FWHM}`.
        """
        return self.fwhm*2.


class OnSkySersic(functional_models.Sersic2D, Source):
    """
    An on-sky Sersic distribution.

    Args:
        sb_eff (scalar-like):
            The surface brightness at 1 effective (half-light) radius.
        r_eff (scalar-like):
            The effective (half-light) radius in *arcseconds*.
        n (scalar-like):
            The Sersic index.
        center (scalar-like, optional):
            The coordinates of the Sersic center in *arcseconds*
            relative to the image center.
        ellipticity (scalar-like, optional):
            The ellipticity (1-b/a) of an elliptical Sersic
            distribution.
        position_angle (scalar-like, optional):
            The position angle for the elliptical Sersic distribution,
            defined as the angle from N through E.  The coordinate
            system is defined with positive offsets (in RA) toward the
            east, meaning lower pixel indices.
        sampling (scalar-like, optional):
            Sampling of a generated map in arcseconds per pixel.
            Default is set by :func:`minimum_sampling`.
        size (scalar-like, optional):
            Size of the image to generate of the distribution in
            *arceconds* along one of the axes.  The map is square.
            Default is defined by :func:`minimum_size`.
        unity_integral (:obj:`bool`, optional):
            Renormalize the distribution so that the integral is unity.
    """
    def __init__(self, sb_eff, r_eff, n, center=None, ellipticity=None, position_angle=None,
                 sampling=None, size=None, unity_integral=False):

        self.position_angle = 0 if position_angle is None else position_angle
        super(OnSkySersic, self).__init__(amplitude=sb_eff, r_eff=r_eff, n=n,
                                          x_0=0 if center is None else center[0],
                                          y_0=0 if center is None else center[1],
                                          ellip=ellipticity,
                                          theta=numpy.radians(90-self.position_angle))

        self.bn = special.gammaincinv(2. * self.n, 0.5)
        self.integral = self.get_integral()
        
        if unity_integral:
            self.amplitude /= self.integral
            self.integral = self.get_integral()

        # Set the map if requested
        if size is not None:
            self.make_map(sampling=sampling, size=size)

    def get_integral(self):
        self.bn = special.gammaincinv(2. * self.n, 0.5)
        return 2 * numpy.pi * self.n * numpy.exp(self.bn) * self.amplitude \
                            * numpy.square(self.r_eff) * special.gamma(2*self.n) \
                            * numpy.power(self.bn, -2*self.n)

    def minimum_sampling(self):
        r"""
        Return the minimum sampling in arcseconds per pixels.  Currently
        :math:`R_{\rm eff}/3`.
        """
        return self.r_eff/3.

    def minimum_size(self):
        r"""
        The minimum size that should be used for the distribution map in
        arcseconds.  Currently :math:`3\ R_{\rm eff}`.
        """
        return self.r_eff*3

# TODO: Add an input image distribution
#class OnSkyImage:
#    def __init__(self, fitsfile):

# TODO: Add a Moffat distribution

class OnSkySource(Source):
    """
    Container class for an on-sky source convolved with the seeing disk.

    .. todo::
        - Allow the seeing kernel to be provided directly instead of
          always adopting a Gaussian with a provided FWHM.

    Args:
        seeing (:obj:`float`, object):

            The FWHM of a Gaussian seeing distribution in arcseconds or an object with a 



            The sampling of the seeing kernel is always FWHM/20 unless
            adjusted by the intrnsic source object.
        intrinsic (:obj:`float`, object):
            The intrinsic surface brightness distribution of the source.
            Can be the total flux of a point source (in, e.g., 10^-17
            erg/s/cm^2/angstrom) or an object.  For the latter, the
            object must have a `__call__` method that takes the on-sky x
            and y position and returns the surface brightness.  Optional
            methods/attributes of the object that are used if available
            are:
                - a `minimum_sampling` method can overwrite the default
                  of FWHM/20 for the pixel sampling.
                - an `integral` attribute can be used to set the
                  integral of the source to a specific value; if not
                  available, the integral is determined by directly
                  integrating the object over the image size using the
                  `__call__` method.
        size (scalar-like, optional):
            The size in arcseconds for the image along one axis
            generated for the source.  The map is square.
    """
#        offset (array-like, optional):
#            The x and y offset to apply (in sky-right arcseconds) via
#            the convolution with the seeing distribution.
    def __init__(self, seeing, intrinsic, sampling=None, size=None): #, offset=None):

        self.seeing_fwhm = seeing_fwhm           # arcsec

        # Offset applied via the convolution
        self.offset = numpy.array([0,0]) if offset is None else numpy.asarray(offset)
        if len(self.offset) != 2:
            raise ValueError('Offset must be a 2-element array.')

        # Sampling in arcsec / pixel
        self.sampling = self.seeing_fwhm/20 if sampling is None else sampling
        try:
            self.sampling = min(intrinsic.minimum_sampling(), self.sampling)
        except AttributeError:
            pass

        # Get the canvas size
        self.size = int((self.seeing_fwhm*3 + numpy.sqrt(numpy.sum(numpy.square(self.offset)))
                            if size is None else size)/self.sampling)+1
        try:
            self.size = max(intrinsic.size, self.size)
        except AttributeError:
            pass

        if self.size % 2 == 0:
            # Size should be odd
            self.size += 1

        # Sample the seeing disk
        kernel = OnSkyGaussian(self.seeing_fwhm/self.sampling, size=self.size)
                                #, center=self.offset/self.sampling)

        # Get coordinates in arcsec
        self.x = kernel.x*self.sampling
        self.y = kernel.y*self.sampling

        # Get the image data
        if isinstance(intrinsic, (int, float)):
            self.data = intrinsic*kernel.data
            self.integral = intrinsic
        else:
            self.data = signal.fftconvolve(intrinsic(self.x, self.y), kernel.data, mode='same')
            try:
                self.integral = intrinsic.integral
            except AttributeError:
                self.integral = numpy.square(self.sampling) * numpy.sum(intrinsic(self.x, self.y))

    def minimum_sampling(self):
        pass

    def minimum_size(self):
        pass

    def make_map(self, sampling=None, size=None):
        """Generate a map of the distribution."""
        # Get the map x and y coordinates
        _size = self.minimum_size() if size is None else size
        _samp = self.minimum_sampling() if sampling is None else sampling
        pixsize = numpy.ceil(_size/_samp).astype(int)
        if pixsize % 2 == 0:
            pixsize += 1
        c = (pixsize-1)*numpy.linspace(-0.5,0.5,pixsize)*_samp
        self.x, self.y = numpy.meshgrid(c[::-1],c)
        # Sample it
        self.data = self.__call__(self.x, self.y)

