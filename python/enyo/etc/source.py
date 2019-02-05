#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Define an on-sky source distribution
"""
import numpy

from scipy import signal, special, interpolate

from astropy.modeling import functional_models

# TODO: Need to learn how to use abstract classes!  Inherit from
# numpy.ndarray?
class Source:
    """
    This is an abstract class an cannot be instantiated on it's own!

    Attributes:
        x (vector): 1D vector with x coordinates
        X (array): 2D map of x coordinates
        y (vector): 1D vector with y coordinates
        Y (array): 2D map of y coordinates
    """
    def __init__(self):
        self.x = None
        self.X = None
        self.y = None
        self.Y = None
        self.data = None
        self.sampling = None
        self.size = None

    def __call__(self, x, y):
        pass
    
    def minimum_sampling(self):
        pass

    def minimum_size(self):
        pass

    def make_map(self, sampling=None, size=None):
        """Generate a map of the distribution."""
        if sampling is None and self.sampling is None:
            self.sampling = self.minimum_sampling()
        elif sampling is not None:
            self.sampling = sampling
            
        if size is None and self.size is None:
            self.size = self.minimum_size()
        elif size is not None:
            self.size = size

        pixsize = numpy.ceil(self.size/self.sampling).astype(int)
        if pixsize % 2 == 0:
            pixsize += 1
        self.y = (pixsize-1)*numpy.linspace(-0.5,0.5,pixsize)*self.sampling
        self.x = self.y.copy()[::-1]

        # Sample it
        self.X, self.Y = numpy.meshgrid(self.x, self.y)
        self.data = self.__call__(self.X, self.Y)

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
            The coordinates of the Gaussian center in *arcseconds*.
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

        # Set the map sampling and size
        self.sampling = sampling
        self.size = size

        # Set the map if requested
        if sampling is not None or size is not None:
            self.make_map()

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

        self.bn = None
        self.integral = self.get_integral()
        
        if unity_integral:
            self.amplitude /= self.integral
            self.integral = self.get_integral()

        # Set the map sampling and size
        self.sampling = sampling
        self.size = size

        # Set the map if requested
        if sampling is not None or size is not None:
            self.make_map()

    def get_integral(self):
        """
        The integral of the Sersic profile projected on the sky.  Note
        the (1-ellipticity) factor.
        """
        self.bn = special.gammaincinv(2. * self.n, 0.5)
        return 2 * numpy.pi * self.n * numpy.exp(self.bn) * self.amplitude \
                            * numpy.square(self.r_eff) * (1-self.ellip) \
                            * special.gamma(2*self.n) * numpy.power(self.bn, -2*self.n)

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

    Unlike the other `Source`s, this requires a map to work.

    Args:
        seeing (:obj:`float`, :class:`Source`):
            The FWHM of a Gaussian seeing distribution in arcseconds or
            an object used to define the seeing kernel directly.  If a
            float is provided, the sampling of the Gaussian seeing
            kernel is set by `OnSkyGaussian.minimum_sampling` unless
            adjusted by the intrinsic source object or the `sampling`
            keyword.  If a :class:`Source` object, the object is used to
            generate a map of the source surface brightness
            distribution.  The integral of the seeing kernel should be
            unity!
        intrinsic (:obj:`float`, :class:`Source`):
            The intrinsic surface brightness distribution of the source.
            Can be the total flux of a point source (in, e.g., 10^-17
            erg/s/cm^2/angstrom) or an object.  If a :class:`Source`
            object, the object is used to generate a map of the source
            surface brightness distribution.
        sampling (scalar-like, optional):
            Sampling of a generated map in arcseconds per pixel.
            Default is set by :func:`minimum_sampling`.
        size (scalar-like, optional):
            Size of the image to generate of the distribution in
            *arceconds* along one of the axes.  The map is square.
            Default is defined by :func:`minimum_size`.
    """
    def __init__(self, seeing, intrinsic, sampling=None, size=None):

        # The seeing kernel
        self.seeing = OnSkyGaussian(seeing) if isinstance(seeing, float) else seeing
        # TODO: Make sure seeing object has unity integral!

        # The intrinsic source distribution
        self.intrinsic = intrinsic

        # Get the sampling
        self.sampling = self.minimum_sampling() if sampling is None else sampling
        self.size = self.minimum_size() if size is None else size

        # Make the map
        self.interp = None
        self.make_map()

    def minimum_sampling(self):
        # Sampling in arcsec / pixel
        sampling = self.seeing.minimum_sampling()
        try:
            # Try using `intrinsic` as an object
            sampling = min(self.intrinsic.minimum_sampling(), sampling)
        except AttributeError:
            pass
        return sampling

    def minimum_size(self):
        # Size in arcsec
        size = self.seeing.minimum_size()
        try:
            # Try using `intrinsic` as an object
            size = max(self.intrinsic.minimum_size(), size)
        except AttributeError:
            pass
        return size

    def make_map(self, sampling=None, size=None):
        if sampling is None and self.sampling is None:
            self.sampling = self.minimum_sampling()
        elif sampling is not None:
            self.sampling = sampling
            
        if size is None and self.size is None:
            self.size = self.minimum_size()
        elif size is not None:
            self.size = size

        # Build the on-sky source distribution
        self.seeing.make_map(sampling=self.sampling, size=self.size)
        self.x = self.seeing.x
        self.X = self.seeing.X
        self.y = self.seeing.y
        self.Y = self.seeing.Y
        try:
            self.intrinsic.make_map(sampling=self.sampling, size=self.size)
            self.data = signal.fftconvolve(self.intrinsic.data, self.seeing.data, mode='same')
        except AttributeError:
            self.data = self.intrinsic*self.seeing.data

        # Get the integral
        try:
            self.integral = self.intrinsic.integral
        except AttributeError:
            self.integral = numpy.square(self.sampling) * numpy.sum(self.data)

        # Prep for interpolation
        self.interp = interpolate.interp2d(self.x, self.y, self.data, bounds_error=True)
#        self.interp = interpolate.RectBivariateSpline(self.x, self.y, self.data)

    def __call__(self, x, y):
        """
        Sample the source.  This interpolates the pre-calcuated source
        at the requested coordinate.  A `ValueError` will be thrown (see
        `scipy.interpolate.interp2d`) if the coordinate is outsize the
        bounds of the calculated map.
        """
        return self.interp(x,y)


