#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Define apertures to use for on-sky integrations.

Requires the python packages shapely_ and rtree_; the latter is python
wrapper for and requires the C++ library libspatialindex_.

.. _shapely: https://shapely.readthedocs.io/en/stable/
.. _rtree: http://toblerity.org/rtree/
.. _libspatialindex: https://libspatialindex.github.io/

.. _shapely.buffer: https://shapely.readthedocs.io/en/stable/manual.html#object.buffer

"""

import os
import numpy
import time

from rtree import index
from shapely.geometry import Point, asPolygon
from shapely.geometry.polygon import Polygon
from shapely.affinity import rotate
import shapely.prepared

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
        if numpy.any(numpy.diff(numpy.diff(x)) > 0):
            raise ValueError('X coordinates are not regular to numerical precision.')
        if numpy.any(numpy.diff(numpy.diff(y)) > 0):
            raise ValueError('Y coordinates are not regular to numerical precision.')

        # Grid shape
        nx = len(x)
        ny = len(y)

        if method == 'whole':
            # Only include whole pixels
            x,y = map(lambda x : x.ravel(), numpy.meshgrid(x, y)) #, indexing='ij'))
            return numpy.array(list(map(lambda x: self.shape.contains(Point(x[0],x[1])),
                                        zip(x,y)))).reshape(nx,ny).astype(int)

        elif method == 'fractional':
            # Allow for fractional pixels by determining the overlap
            # between the shape and each grid cell

            # Build the cell polygons
            cells = Aperture._get_grid_tree(x, y)[0]

            # Construct a grid with the fractional area covered by the
            # aperture
            return numpy.array(list(map(lambda x: self.shape.intersection(x).area,
                                        cells))).reshape(nx,ny)

        raise ValueError('Unknown grid_overlap method {0}.'.format(method))

        # OLD and slow
        cells, tree = Aperture._get_grid_tree(x, y, fast=False)
        alpha = numpy.zeros((nx,ny), dtype=float)
        for k in tree.intersection(self.shape.bounds):
            i = k//ny
            j = k - i*ny
            alpha[i,j] = cells[k].intersection(self.shape).area
        return alpha

    @staticmethod
    def _get_grid_tree(x, y, fast=True):
        r"""
        Construct the polygons and RTree for searching.

        The list of polygons follows array index order.  I.e., polygon
        :math:`k` is the cell at location :math:`(i,j)`, where::

        .. math::
            
            i = k//ny
            j = k - i*ny 

        Args:
            x (array-like):
                The list of x coordinates for the grid.  Must be
                linearly spaced.
            y (array-like):
                The list of y coordinates for the grid.  Must be
                linearly spaced.
            fast (:obj:`bool`, optional):
                Skip the construction of the search tree.  The tree is
                returned as None.
        
        Returns:
            Two objects are returned:
                - a list of shapely.geometry.polygon.Polygon objects, on
                  per grid cell
                - an rtree index object for quickly finding the
                  intersection between these grid cells and the aperture
                  polygon.
        """
        # Get the cell size
        dx = abs(x[1]-x[0])
        dy = abs(y[1]-y[0])

        # Construct the grid
        x,y = map(lambda x : x.ravel(), numpy.meshgrid(x, y)) #, indexing='ij'))

        # Construct the polygons
        cx = x[:,None] + (numpy.array([-0.5,0.5,0.5,-0.5])*dx)[None,:]
        cy = y[:,None] + (numpy.array([-0.5,-0.5,0.5,0.5])*dy)[None,:]
        polygons = [asPolygon(p) for p in 
                                numpy.append(cx, cy, axis=1).reshape(-1,2,4).transpose(0,2,1)]
        if fast:
            return polygons, None

        # Build the polygon tree
#        if use_strtree:
#            return polygons, STRtree(polygons)
        tree = index.Index()
        for i, p in enumerate(polygons):
            tree.insert(i,p.bounds)      
        return polygons, tree

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
        x = numpy.array([-width/2, width/2])+cx
        y = numpy.array([-length/2, length/2])+cy
        square = asPolygon(numpy.append(numpy.roll(numpy.repeat(x,2),-1),
                                        numpy.repeat(y,2)).reshape(2,4).T)
        super(SlitAperture, self).__init__(rotate(square, rotation))


