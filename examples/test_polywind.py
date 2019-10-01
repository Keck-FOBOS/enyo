
import time

import numpy
from matplotlib import pyplot

from mangadap.util.geometry import point_inside_polygon

def polygon_winding_number_2(polygon, point):
    """
    Determine the winding number of a 2D polygon about a point.  The
    code does **not** check if the polygon is simple (no interesecting
    line segments).  Algorithm taken from Numerical Recipies Section
    21.4.

    Args:
        polygon (numpy.ndarray): An Nx2 array containing the x,y
            coordinates of a polygon.  The points should be ordered
            either counter-clockwise or clockwise.
        point (numpy.ndarray): A 2-element array defining the x,y
            position of the point to use as a reference for the winding
            number.

    Returns:
        int: Winding number of `polygon` w.r.t. `point`

    Raises:
        ValueError: Raised if `polygon` is not 2D, if `polygon` does not
            have two columns, or if `point` is not a 2-element array.
    """
    # Check input shape is for 2D only
    if len(polygon.shape) != 2:
        raise ValueError('Polygon must be an Nx2 array.')
    if polygon.shape[1] != 2:
        raise ValueError('Polygon must be in two dimensions.')
    _point = numpy.atleast_2d(point)
    if _point.shape[1] != 2:
        raise ValueError('Point must contain two elements.')

    # Get the winding number
    nvert = polygon.shape[0]
    np = _point.shape[0]

    dl = numpy.roll(polygon, 1, axis=0)[None,:,:] - _point[:,None,:]
    dr = polygon[None,:,:] - point[:,None,:]
    dx = dl[:,:,0]*dr[:,:,1] - dl[:,:,1]*dr[:,:,0]

    indx_l = dl[:,:,1] > 0
    indx_r = dr[:,:,1] > 0

    wind = numpy.zeros((np, nvert), dtype=int)
    wind[indx_l & numpy.invert(indx_r) & (dx < 0)] = -1
    wind[numpy.invert(indx_l) & indx_r & (dx > 0)] = 1

    return numpy.sum(wind, axis=1)[0] if point.ndim == 1 else numpy.sum(wind, axis=1)

def point_inside_polygon_2(polygon, point):
    return numpy.absolute(polygon_winding_number_2(polygon, point)) == 1


def point_inside_polygon_3(polygon, point):
    # Check input shape is for 2D only
    if len(polygon.shape) != 2:
        raise ValueError('Polygon must be an Nx2 array.')
    if polygon.shape[1] != 2:
        raise ValueError('Polygon must be in two dimensions.')
    _point = numpy.atleast_2d(point)
    if _point.shape[1] != 2:
        raise ValueError('Point must contain two elements.')
    
    from shapely.geometry import Point, Polygon
    poly = Polygon(polygon.tolist())
    import pdb
    pdb.set_trace()

    return numpy.array([ poly.contains(Point(p)) for p in _point])


def main():

    polygon = numpy.array([ [0,0], [2,0], [2,1], [1,1], [1,2], [0,2]]).astype(float)

    pnts = numpy.random.uniform(low=-1, high=3, size=(100,2))

    t = time.perf_counter()
    for i in range(5):
        indx_o = point_inside_polygon(polygon, pnts)
    print(time.perf_counter() - t)

    t = time.perf_counter()
    for i in range(5):
        indx_2 = point_inside_polygon_2(polygon, pnts)
    print(time.perf_counter() - t)

    t = time.perf_counter()
    for i in range(5):
        indx_3 = point_inside_polygon_3(polygon, pnts)
    print(time.perf_counter() - t)

    print(numpy.all(indx_o.astype(int) - indx_2.astype(int) == 0))
    print(numpy.all(indx_o.astype(int) - indx_3.astype(int) == 0))

#    pyplot.plot(numpy.append(polygon[:,0], polygon[0,0]),
#                numpy.append(polygon[:,1], polygon[0,1]), color='C3')
#    pyplot.scatter(pnts[:,0], pnts[:,1], color='k', marker='.', lw=0, s=30)
#    pyplot.scatter(pnts[indx_2,0], pnts[indx_2,1], color='C3', marker='.', lw=1,
#                   facecolor='none', s=100)
#    pyplot.show()


if __name__ == '__main__':
    main()


