#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Define an on-sky source distribution
"""

import numpy

from scipy import signal, special

from astropy.modeling import functional_models

class OnSkyGaussian(functional_models.Gaussian2D):
    """
    center and sigma are in pixel units

    coordinates are sky-right (+x to east/left)

    PA is angle from N through E

    map is always square
    """
    def __init__(self, fwhm, size=None, center=None, ellipticity=None, position_angle=None):

        self.fwhm = fwhm
        self.center = [0,0] if center is None else center
        self.ellipticity = 0 if ellipticity is None else ellipticity
        self.position_angle = 0 if position_angle is None else position_angle

        major_sigma = self.fwhm / numpy.sqrt(8*numpy.log(2))
        minor_sigma = major_sigma * (1-self.ellipticity)
        super(OnSkyGaussian, self).__init__(amplitude=1/(2*major_sigma*minor_sigma*numpy.pi),
                                            x_mean=self.center[0], y_mean=self.center[1],
                                            x_stddev=minor_sigma, y_stddev=major_sigma,
                                            theta=-numpy.radians(self.position_angle))

        self.integral = 1.0

        self.x = None
        self.y = None
        self.data = None
        if size is not None:
            self.make_map(size)

    def minimum_sampling(self):
        return self.fwhm/2

    def minimum_size(self):
        return self.fwhm*2

    def make_map(self, size):
        # Get the map x and y coordinates
        c = (size-1)*numpy.linspace(-0.5,0.5,size)
        self.x, self.y = numpy.meshgrid(c[::-1],c)
        self.data = self.__call__(self.x, self.y)

    def __getitem__(self, s):
        if self.data is None:
            raise ValueError('Distribution data is not defined!')
        return self.data[s]

    @property
    def shape(self):
        return () if self.data is None else self.data.shape


class OnSkySersic(functional_models.Sersic2D):
    """
    center and r_eff are in pixels if size is not None

    coordinates are sky-right (+x to east/left)

    PA is angle from N through E

    map is always square
    """
    def __init__(self, sb_eff, r_eff, n, size=None, center=None, ellipticity=None,
                 position_angle=None, unity_integral=False):

        self.sb_eff = sb_eff
        self.center = [0,0] if center is None else center
        self.ellipticity = 0 if ellipticity is None else ellipticity
        self.position_angle = 0 if position_angle is None else position_angle
        super(OnSkySersic, self).__init__(amplitude=self.sb_eff, r_eff=r_eff, n=n,
                                          x_0=self.center[0], y_0=self.center[1],
                                          ellip=self.ellipticity,
                                          theta=numpy.radians(90-self.position_angle))

        self.bn = special.gammaincinv(2. * self.n, 0.5)
        self.integral = 2 * numpy.pi * self.n * numpy.exp(self.bn) * self.sb_eff \
                            * numpy.square(self.r_eff) * special.gamma(2*self.n) \
                            / numpy.power(self.bn, 2*self.n)

        if unity_integral:
            self.sb_eff /= self.integral
            self.integral = 1

        self.x = None
        self.y = None
        self.data = None
        if size is not None:
            self.make_map(size)

    def minimum_sampling(self):
        return self.r_eff/3

    def minimum_size(self):
        return self.r_eff*3

    def make_map(self, size):
        # Get the map x and y coordinates
        c = (size-1)*numpy.linspace(-0.5,0.5,size)
        self.x, self.y = numpy.meshgrid(c[::-1],c)
        self.data = self.__call__(self.x, self.y)

    def __getitem__(self, s):
        if self.data is None:
            raise ValueError('Distribution data is not defined!')
        return self.data[s]

    @property
    def shape(self):
        return () if self.data is None else self.data.shape


# TODO: Add an input image distribution
#class OnSkyImage:
#    def __init__(self, fitsfile):


class OnSkySource:
    """
    intrinsic can be a flux (10^-17 erg/s/cm^2/angstrom) or an object
    that can be called with the x and y position and return the surface
    brightness.  It cannot be None.
    """
    def __init__(self, seeing_fwhm, intrinsic, size=None, offset=None):
        # Determine the sampling
        self.seeing_fwhm = seeing_fwhm           # arcsec
        self.sampling = self.seeing_fwhm/20      # arcsec / pixel
        if intrinsic is not None:
            minsampling = intrinsic.minimum_sampling()  # arcsec/pixel
            self.sampling = min(minsampling, self.sampling)

        self.offset = numpy.array([0,0]) if offset is None else numpy.asarray(offset)

        # Get the canvas size
        self.size = int((self.seeing_fwhm*3 + numpy.sqrt(numpy.sum(numpy.square(self.offset)))
                            if size is None else size)/self.sampling)+1

        if self.size % 2 == 0:
            # Size should be odd
            self.size += 1

        # Sample the seeing disk
        kernel = OnSkyGaussian(self.seeing_fwhm/self.sampling, self.size,
                               center=self.offset/self.sampling)
        self.x = kernel.x*self.sampling
        self.y = kernel.y*self.sampling

        if isinstance(intrinsic, (int, float)):
            self.data = intrinsic*kernel.data
            self.integral = intrinsic
        else:
            self.data = signal.fftconvolve(intrinsic(self.x, self.y), kernel.data, mode='same')
            self.integral = intrinsic.integral

    def __getitem__(self, s):
        if self.data is None:
            raise ValueError('Distribution data is not defined!')
        return self.data[s]

    @property
    def shape(self):
        return () if self.data is None else self.data.shape


