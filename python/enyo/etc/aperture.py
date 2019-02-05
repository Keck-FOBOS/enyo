#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Define apertures to use for on-sky integrations.
"""

#Requires the python packages shapely_ and rtree_; the latter is python
#wrapper for and requires the C++ library libspatialindex_.
#
#.. _shapely: https://shapely.readthedocs.io/en/stable/
#.. _rtree: http://toblerity.org/rtree/
#.. _libspatialindex: https://libspatialindex.github.io/
#
#.. _shapely.buffer: https://shapely.readthedocs.io/en/stable/manual.html#object.buffer

import os
import numpy
import time

#from rtree import index
#from shapely.geometry.polygon import Polygon
#import shapely.prepared
from shapely.geometry import Point, asPolygon
from shapely.affinity import rotate

class Aperture:
    """
    Abstract class for a general aperture shape.

    Args:
        shape (shapely.geometry.base.BaseGeometry):
            A shape object from the Shapely python package.
    """
    def __init__(self, shape):
#        self.shape = shapely.prepared.prep(shape)
        self.shape = shape

    def grid_overlap(self, x, y, method='whole'):
        """
        Compute the ovelapping area between the aperture and the cells
        in a regular grid.

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
                      aperture is set to 1.  All others set to 0.
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
                                    else numpy.finfo(x.dtype).eps*10
        minimum_y_difference = 0 if numpy.issubdtype(y.dtype, numpy.integer) \
                                    else numpy.finfo(y.dtype).eps*10
        if numpy.any(numpy.absolute(numpy.diff(numpy.diff(x))) > minimum_x_difference):
            raise ValueError('X coordinates are not regular to numerical precision.')
        if numpy.any(numpy.absolute(numpy.diff(numpy.diff(y))) > minimum_y_difference):
            raise ValueError('Y coordinates are not regular to numerical precision.')

        # Grid shape
        nx = len(x)
        ny = len(y)

        if method == 'whole':
            # Only include whole pixels
            X,Y = map(lambda x : x.ravel(), numpy.meshgrid(x, y))
            return numpy.array(list(map(lambda x: self.shape.contains(Point(x[0],x[1])),
                                        zip(X,Y)))).reshape(ny,nx).astype(int)

        elif method == 'fractional':
            # Allow for fractional pixels by determining the overlap
            # between the shape and each grid cell

            # Build the cell polygons
            cells, sx, ex, sy, ey = self._overlapping_grid_polygons(x, y)

            # Construct a grid with the fractional area covered by the
            # aperture
            img = numpy.zeros((len(y), len(x)), dtype=float)
            img[sy:ey,sx:ex] = numpy.array(list(map(lambda x: self.shape.intersection(x).area,
                                                cells))).reshape(ey-sy,ex-sx)
            return img
            
#            return numpy.array(list(map(lambda x: self.shape.intersection(x).area,
#                                        cells))).reshape(nx,ny)

        raise ValueError('Unknown grid_overlap method {0}.'.format(method))

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

#        if fast:
#            return polygons, None,

#        # Build the polygon tree
##        if use_strtree:
##            return polygons, STRtree(polygons)
#        tree = index.Index()
#        for i, p in enumerate(polygons):
#            tree.insert(i,p.bounds)      
#        return polygons, tree

    @property
    def area(self):
        return self.shape.area
        

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
    """
    def __init__(self, cx, cy, d, resolution=None):
        self.center = (cx,cy)
        self.diamter = d
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
    """
    def __init__(self, cx, cy, width, length, rotation=0.):
        self.center = (cx,cy)
        self.width = width
        self.length = length
        x = numpy.array([-width/2, width/2])+cx
        y = numpy.array([-length/2, length/2])+cy
        square = asPolygon(numpy.append(numpy.roll(numpy.repeat(x,2),-1),
                                        numpy.repeat(y,2)).reshape(2,4).T)
        super(SlitAperture, self).__init__(rotate(square, rotation))


