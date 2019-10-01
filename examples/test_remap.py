
import numpy
from matplotlib import pyplot

from enyo.etc import spectrographs

tmtb = spectrographs.TMTWFOSBlueOpticalModel()

test_img = numpy.zeros((100,50), dtype=float)
wave0 = 3110.
pixelscale = 0.05153458543289052
dispscale = 15 #0.1995
test_img[20,:] = 1
test_img[60,:] = 1
test_img[:,10] = 1
test_img[:,30] = 1
#pyplot.imshow(test_img, origin='lower', interpolation='nearest', aspect='auto')
#pyplot.show()

spec, spec0, spat0 \
    = tmtb.project_2d_spectrum(test_img, pixelscale, wave0, dispscale, field_coo=numpy.array([-3,0.5]))

print(spec0, spat0, spec.shape)

