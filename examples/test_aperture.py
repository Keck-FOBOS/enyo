
import numpy
from matplotlib import pyplot
from enyo.etc.aperture import FiberAperture

ap = FiberAperture(0., 0., 1., resolution=100)

x = numpy.linspace(-1.,1.,101)
y = numpy.linspace(-1.,1.,101)
import time
t = time.perf_counter()
img = ap.response(x,y,method='fractional')
print(time.perf_counter()-t)
sampling = x[1]-x[0]
print(numpy.sum(img*numpy.square(sampling)))
print(ap.area)

pyplot.imshow(img, origin='lower', interpolation='nearest')
pyplot.show()

exit()

x = numpy.arange(3)
y = numpy.arange(2)
dx = 1
dy = 1

# Construct the grid
X,Y = map(lambda x : x.ravel(), numpy.meshgrid(x, y))

# Construct the polygons
cx = X[:,None] + (numpy.array([-0.5,0.5,0.5,-0.5])*dx)[None,:]
cy = Y[:,None] + (numpy.array([-0.5,-0.5,0.5,0.5])*dy)[None,:]

boxes = numpy.append(cx, cy, axis=1).reshape(-1,2,4).transpose(0,2,1)


