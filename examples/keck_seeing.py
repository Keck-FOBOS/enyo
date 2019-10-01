
#    - Effect of seeing distribution
#    - Number of fibers in pseudo slit length
#    - Convert S/N to survey time

import numpy

from scipy import interpolate

from matplotlib import pyplot

seeing, count = numpy.genfromtxt('keck_seeing.db').T

print(seeing)
exit()

seeing = numpy.append([0.25], seeing)
count = numpy.append([0.0], count)
integ = numpy.cumsum(count)/numpy.sum(count)

srng = [numpy.amin(seeing)-0.025, numpy.amax(seeing)+0.025]

ss = numpy.linspace(*srng, 100)

# Preserves monotonically increasing data
modmono = interpolate.PchipInterpolator(seeing+0.025, integ)(ss)

#pyplot.step(seeing, integ, where='mid')
#pyplot.plot(ss, modmono)
#pyplot.show()

growth_invert = interpolate.interp1d(modmono, ss)

sample = numpy.random.uniform(size=100000)

pyplot.hist(growth_invert(sample), bins=4*len(count), range=srng, density=True)
pyplot.step(seeing, count/numpy.sum(count)/0.05, where='mid')
pyplot.show()
exit()

#print(modmono)
#print(numpy.diff(modmono))


for o in [9, 11]:
    p = numpy.polynomial.Legendre.fit(seeing, integ, o)
    ss, mod = p.linspace()
    pyplot.plot(seeing, p(seeing)-integ)
    pyplot.scatter(seeing, p(seeing)-integ, marker='.', s=60, lw=0)
#    pyplot.plot(ss, mod)

pyplot.show()


