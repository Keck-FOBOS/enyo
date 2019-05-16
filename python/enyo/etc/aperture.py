#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Define apertures to use for on-sky integrations.
"""

import os
import numpy
import time

from scipy import signal

from shapely.geometry import Point, asPolygon
from shapely.affinity import rotate

class Aperture:
    """
    Abstract class for a general aperture shape.

    .. todo:
        - limit the calculation of the polygon overlap to where the
        corners of the grid cells cross the boundary of the aperture.

    Args:
        shape (shapely.geometry.base.BaseGeometry):
            A shape object from the Shapely python package.
    """
    def __init__(self, shape):
#        self.shape = shapely.prepared.prep(shape)
        self.shape = shape

    def response(self, x, y, method='fractional'):
        """
        Compute the response function of the aperture to the sky over a
        regular grid.

        The integral of the returned map is normalized to the area of
        the aperture.

        Args:
            x (array-like):
                The list of x coordinates for the grid.  Must be
                linearly spaced.
            y (array-like):
                The list of y coordinates for the grid.  Must be
                linearly spaced.
            method (:obj:`str`, optional):
                Method used to construct the overlap grid.  Options
                are::
                    - 'whole': Any grid cell with its center inside the
                      aperture is set to the area of the grid cell.  All
                      others set to 0.
                    - 'fractional': Perform the detailed calculation of
                      the fraction of each grid-cell within the
                      aperture.
    
        Returns:
            numpy.ndarray: An array with shape (nx, ny) with the
            fraction of each grid cell covered by the aperture.

        Raises:
            ValueError:
                Raised if the provided arguments are not regularly
                spaced, or if there aren't at least 2 grid points in
                each dimension.
        """
        # Check input
        if len(x) < 2 or len(y) < 2:
            raise ValueError('Must provide at least 2 points per grid point.')
        minimum_x_difference = 0 if numpy.issubdtype(x.dtype, numpy.integer) \
                                    else numpy.finfo(x.dtype).eps*100
        minimum_y_difference = 0 if numpy.issubdtype(y.dtype, numpy.integer) \
                                    else numpy.finfo(y.dtype).eps*100
        if numpy.any(numpy.absolute(numpy.diff(numpy.diff(x))) > minimum_x_difference):
            raise ValueError('X coordinates are not regular to numerical precision.')
        if numpy.any(numpy.absolute(numpy.diff(numpy.diff(y))) > minimum_y_difference):
            raise ValueError('Y coordinates are not regular to numerical precision.')

        # Grid shape
        nx = len(x)
        ny = len(y)

        # Get the cell size
        dx = abs(x[1]-x[0])
        dy = abs(y[1]-y[0])

        cell_area = dx*dy

        if method == 'whole':
            # Only include whole pixels
            X,Y = map(lambda x : x.ravel(), numpy.meshgrid(x, y))
            img = numpy.array(list(map(lambda x: self.shape.contains(Point(x[0],x[1])),
                                        zip(X,Y)))).reshape(ny,nx).astype(int)/cell_area
            return img * (self.area/numpy.sum(img)/cell_area)

        elif method == 'fractional':
            # Allow for fractional pixels by determining the overlap
            # between the shape and each grid cell

            # Build the cell polygons
            cells, sx, ex, sy, ey = self._overlapping_grid_polygons(x, y)

            # Construct a grid with the fractional area covered by the
            # aperture
            img = numpy.zeros((len(y), len(x)), dtype=float)
            img[sy:ey,sx:ex] = numpy.array(list(map(lambda x: self.shape.intersection(x).area,
                                                cells))).reshape(ey-sy,ex-sx)/cell_area
            return img * (self.area/numpy.sum(img)/cell_area)

        raise ValueError('Unknown response method {0}.'.format(method))

#        # OLD and slow
#        cells, tree = Aperture._get_grid_tree(x, y, fast=False)
#        alpha = numpy.zeros((nx,ny), dtype=float)
#        for k in tree.intersection(self.shape.bounds):
#            i = k//ny
#            j = k - i*ny
#            alpha[i,j] = cells[k].intersection(self.shape).area
#        return alpha

    def _overlapping_grid_polygons(self, x, y):
        r"""
        Construct the list grid-cell polygons (rectangles) that are
        expected to overlap the aperture.

        The list of polygons follows array index order.  I.e., polygon
        :math:`k` is the cell at location :math:`(j,i)`, where::

        .. math::
            
            j = k//nx
            i = k - j*nx

        Args:
            x (array-like):
                The list of x coordinates for the grid.  Must be
                linearly spaced.
            y (array-like):
                The list of y coordinates for the grid.  Must be
                linearly spaced.
        
        Returns:
            Four objects are returned:
                - A list of shapely.geometry.polygon.Polygon objects, on
                  per grid cell.  Only those grid cells that are
                  expected to overlap the shape's bounding box are
                  included.
                - The starting and ending x index and the starting and
                  ending y index for the returned list of cell polygons.
        """
        # Get the cell size
        dx = abs(x[1]-x[0])
        dy = abs(y[1]-y[0])

        # Find the x coordinates of the grid cells that overlap the shape
        xlim = list(self.shape.bounds[::2])
        if xlim[0] > xlim[1]:
            xlim = xlim[::-1]
        xindx = (x+0.5*dx > xlim[0]) & (x-0.5*dx < xlim[1])
        sx = numpy.arange(len(x))[xindx][0]
        ex = numpy.arange(len(x))[xindx][-1]+1

        # Find the y coordinates of the grid cells that overlap the shape
        ylim = list(self.shape.bounds[1::2])
        if ylim[0] > ylim[1]:
            ylim = ylim[::-1]
        yindx = (y+0.5*dy > ylim[0]) & (y-0.5*dy < ylim[1])
        sy = numpy.arange(len(y))[yindx][0]
        ey = numpy.arange(len(y))[yindx][-1]+3

        # Construct the grid
        X,Y = map(lambda x : x.ravel(), numpy.meshgrid(x[sx:ex], y[sy:ey]))

        # Construct the polygons
        cx = X[:,None] + (numpy.array([-0.5,0.5,0.5,-0.5])*dx)[None,:]
        cy = Y[:,None] + (numpy.array([-0.5,-0.5,0.5,0.5])*dy)[None,:]

        boxes = numpy.append(cx, cy, axis=1).reshape(-1,2,4).transpose(0,2,1)
        polygons = [asPolygon(box) for box in boxes]
        
        return polygons, sx, ex, sy, ey

    @property
    def area(self):
        return self.shape.area

    @property
    def bounds(self):
        return self.shape.bounds

    def integrate_over_source(self, source, response_method='fractional', sampling=None,
                              size=None):
        """
        Integrate a source over the aperture.

        This is done by generating an image of the aperture over the map
        of the source surface-brightness distribution, using
        :func:`Aperture.response`.  The source is expected to already
        have been mapped using its `make_map` function, or one should
        provide `sampling` and `size` values to construct the map inside
        this function.

        See also: :func:`Aperture.map_integral_over_source`.

        .. todo::
            Require source to be a :class:`enyo.etc.source.Source` object?

        Args:
            source (:class:`enyo.etc.source.Source`):
                Source surface-brightness distribution
            response_method (str):
                See `method` argument for :func:`Aperture.response`.
            sampling (:obj:`float`, optional):
                Sampling of the square map in arcsec/pixel.  If not
                None, the source map is reconstructed.
            size (:obj:`float`, optional):
                Size of the square map in arcsec.  If not None, the
                source map is reconstructed.

        Returns:
            float: The integral of the source over the aperture.

        """
        if source.data is None and sampling is None and size is None:
            raise ValueError('Must make a map of the source first.')

        if sampling is not None or size is not None:
            source.make_map(sampling=sampling, size=size)

        aperture_image = self.response(source.x, source.y, method=response_method)
        return numpy.sum(source.data*aperture_image)*numpy.square(source.sampling)

    def map_integral_over_source(self, source, response_method='fractional', sampling=None,
                                 size=None):
        """
        Construct a continuous map of the source integrated over the
        aperture.

        This is done by generating an image of the aperture over the map
        of the source surface-brightness distribution, using
        :func:`Aperture.response`.  The integral of the source over the
        aperture *at any offset position within the map* is calculated
        by convolving the the source distribution and the aperture
        image.

        See also :func:`Aperture.integrate_over_source`.  A single call
        to this function or :func:`Aperture.map_integral_over_source` to
        get the integral with no offset of the aperture are marginally
        different.  However, use of this function is much more efficient
        if you want to calculate the integral of the source over many
        positional offsets of the aperture.
        
        .. todo::
            Require source to be a :class:`enyo.etc.source.Source` object?

        Args:
            source (:class:`enyo.etc.source.Source`):
                Source surface-brightness distribution
            response_method (str):
                See `method` argument for :func:`Aperture.response`.
            sampling (:obj:`float`, optional):
                Sampling of the square map in arcsec/pixel.  If not
                None, the source map is reconstructed.
            size (:obj:`float`, optional):
                Size of the square map in arcsec.  If not None, the
                source map is reconstructed.

        Returns:
            numpy.ndarray: The integral of the source over the aperture
            with the aperture centered at any position in the map.
            The integral with no offset between the image of the
            aperture and the image of the source is::

                cy = source.data.shape[0]//2
                cx = source.data.shape[1]//2
                integral = Aperture.map_integral_over_source(source)[cy,cx]

            which should be identical to::

                integral = Aperture.integrate_over_source(source)

        """
        if source.data is None and sampling is None and size is None:
            raise ValueError('Must make a map of the source first.')

        if sampling is not None or size is not None:
            source.make_map(sampling=sampling, size=size)

        aperture_image = self.response(source.x, source.y, method=response_method)
        return signal.fftconvolve(source.data, aperture_image*numpy.square(source.sampling),
                                  mode='same')
        

class FiberAperture(Aperture):
    """
    Define a fiber aperture.

    Args:
        cx (scalar-like):
            On-sky center X coordinate.
        cy (scalar-like):
            On-sky center Y coordinate.
        d (scalar-like):
            Fiber diameter.  Aperture is assumed to be a circle resolved
            by a set of line segments.
        resolution (:obj:`int`, optional):
            Set the "resolution" of the circle.  Higher numbers mean
            more line segments are used to define the circle, but there
            isn't a 1-1 correspondence.  See shapely.buffer_.  Default
            is to use shapely_ default.

    Attributes:
        center (list):
            Center x and y coordinate.
        diameter (float):
            Fiber diameter
    """
    def __init__(self, cx, cy, d, resolution=None):
        self.center = [cx,cy]
        self.diameter = d
        kw = {} if resolution is None else {'resolution':resolution}
        super(FiberAperture, self).__init__(Point(cx,cy).buffer(d/2, **kw))


class SlitAperture(Aperture):
    """
    Define a slit aperture.

    The orientation of the slit is expected to have the length along the
    y axis and the width along the x axis.  The rotation is
    counter-clockwise in a right-handed Cartesian frame.

    Exactly the same aperture is obtained in the following two calls::

        s = SlitAperture(0., 0., 1, 10)
        ss = SlitAperture(0., 0., 10, 1, rotation=90)

    Args:
        cx (scalar-like):
            On-sky center X coordinate.
        cy (scalar-like):
            On-sky center Y coordinate.
        width (scalar-like):
            Slit width along the unrotated x axis.
        length (scalar-like):
            Slit length along the unrotated y axis.
        rotation (scalar-like):
            Cartesian rotation of the slit in degrees.

    Attributes:
        center (list):
            Center x and y coordinate.
        width (float):
            Slit width
        length (float):
            Slit length
    """
    def __init__(self, cx, cy, width, length, rotation=0.):
        self.center = [cx,cy]
        self.width = width
        self.length = length
        x = numpy.array([-width/2, width/2])+cx
        y = numpy.array([-length/2, length/2])+cy
        square = asPolygon(numpy.append(numpy.roll(numpy.repeat(x,2),-1),
                                        numpy.repeat(y,2)).reshape(2,4).T)
        # rotate() function is provided by shapely.affinity package
        super(SlitAperture, self).__init__(rotate(square, rotation))


