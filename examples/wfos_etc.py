import os

import numpy
from matplotlib import pyplot

from enyo.etc import observe

from enyo.etc import source, efficiency, telescopes, spectrum, extract, aperture

def main():

    # Assume slit is perfectly aligned with center of object

    # TODO: Introduce wavelength dependent seeing?
    mag = 22.               # g-band magnitude
    seeing = 0.6            # arcsec
    slitwidth = 0.75        # arcsec
    slitlength = 5.         # arcsec
    slitrotation = 0.       # degrees

    # Define the slit aperture
    slit = aperture.SlitAperture(0., 0., slitwidth, slitlength, rotation=slitrotation)

    # Sample the source at least 5 times per seeing disk FWHM
    sampling = seeing/5
    # Set the size of the map to be the maximum of 3 seeing disks, or
    # 1.5 times the size of the slit in the cartesian coordinates.
    # First, find the width of the slit shape in x and y (including any
    # rotation).
    dslitx, dslity = numpy.diff(numpy.asarray(slit.bounds).reshape(2,-1), axis=0).ravel()
    size = max(seeing*3, 1.5*dslitx, 1.5*dslity)

    # Use a point source with unity integral (integral of mapped
    # profile will not necessarily be unity)
    star = source.OnSkySource(seeing, 1.0, sampling=sampling, size=size)
    print('Fraction of star flux in mapped image: {0:.3f}'.format(numpy.sum(star.data)
                                                                  * numpy.square(sampling)))

    slit_img = slit.response(star.x, star.y)
    print('Slit area: {0:.2f} (arcsec^2)'.format(slit.area))

    print('Aperture efficiency: {0:.2f}'.format(numpy.sum(slit_img*star.data)
                                                 * numpy.square(sampling)))

    # Use a reference spectrum that's constant; units are 1e-17 erg/s/cm^2/angstrom
    wave = numpy.linspace(3000., 10000., num=7001)
    spec = spectrum.ABReferenceSpectrum(wave)
    g = efficiency.FilterResponse()
    # Rescale to a specified magnitude
    spec.rescale_magnitude(mag, band=g)
    print('Star Magnitude (AB mag): {0:.2f}'.format(spec.magnitude(g)))

    # Sky Spectrum; units are 1e-17 erg/s/cm^2/angstrom/arcsec^2
    sky = spectrum.MaunakeaSkySpectrum()
    sky_mag = sky.magnitude(g)
    print('Sky Surface Brightness (AB mag/arcsec^2): {0:.2f}'.format(sky_mag))


#    obs = observe.Observe()
#
#        (onsky_source_distribution, source_spectrum, sky_spectrum,
#                 atmospheric_throughput, telescope, focal_plane_aperture, system_throughput,
#                 detector, exposure_time, extraction):
#


if __name__ == '__main__':
    main()