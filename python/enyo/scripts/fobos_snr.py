#!/usr/bin/env python3

import os
import time
import warnings
import argparse

from IPython import embed

import numpy

from matplotlib import pyplot, ticker

from enyo.etc import source, efficiency, telescopes, spectrum, extract, aperture, detector
from enyo.etc.observe import Observation


def parse_args(options=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--wavelengths', default=[3100,10000,1e-5], nargs=3, type=float,
                        help='Wavelength grid: start wave, approx end wave, logarithmic step.')
    parser.add_argument('-m', '--mag', default=24., type=float,
                        help='Object total apparent g-band magnitude.')
    parser.add_argument('-z', '--redshift', default=0.0, type=float,
                        help='Redshift of the object, z')
    parser.add_argument('-e', '--emline', default=None, type=str,
                        help='File with emission lines to add to the spectrum.')
    parser.add_argument('-s', '--sersic', default=None, nargs=4, type=float,
                        help='Use a Sersic profile to describe the object surface-brightness '
                             'distribution; order must be effective radius, Sersic index, '
                             'ellipticity (1-b/a), position angle (deg).')
    parser.add_argument('-t', '--time', default=3600., type=float, help='Exposure time (s)')
    parser.add_argument('-f', '--fwhm', default=0.65, type=float,
                        help='On-sky PSF FWHM (arcsec)')
    parser.add_argument('-a', '--airmass', default=1.0, type=float, help='Airmass')
    parser.add_argument('-i', '--ipython', default=False, action='store_true',
                        help='After completing the setup, embed in an IPython session.')
    parser.add_argument('-p', '--plot', default=False, action='store_true',
                        help='Provide a plot of the components of the calculation.')
    parser.add_argument('-u', '--per_ang', default=False, action='store_true',
                        help='Report the S/N per angstrom instead of per resolution element.')

    return parser.parse_args() if options is None else parser.parse_args(options)


def _emission_line_database_dtype():
    return [('name', '<U20'),
            ('flux', float),
            ('restwave', float),
            ('frame', '<U10'),
            ('fwhm', float),
            ('fwhmu', '<U10')]


def read_emission_line_database(dbfile):
    return numpy.genfromtxt(dbfile, dtype=_emission_line_database_dtype())


def get_wavelength_vector(start, end, logstep):
    """
    Get the wavelength vector
    """
    nwave = int((numpy.log10(end)-numpy.log10(start))/logstep + 1)
    return numpy.power(10., numpy.arange(nwave)*logstep + numpy.log10(start))


def get_spectrum(wave, mag, emline_db=None, redshift=0.0, resolution=3500):
    """
    """
    spec = spectrum.ABReferenceSpectrum(wave, resolution=resolution, log=True)
    g = efficiency.FilterResponse()
    spec.rescale_magnitude(g, mag)
    if emline_db is None:
        return spec
    spec = spectrum.EmissionLineSpectrum(wave, emline_db['flux'], emline_db['restwave'],
                                         emline_db['fwhm'], units=emline_db['fwhmu'],
                                         redshift=redshift, resolution=resolution, log=True,
                                         continuum=spec.flux)
    warnings.warn('Including emission lines, spectrum g-band magnitude changed '
                  'from {0} to {1}.'.format(mag, spec.magnitude(g)))
    return spec

def main(args):

    t = time.perf_counter()

    # Constants:
    resolution = 3500.      # lambda/dlambda
    fiber_diameter = 0.8    # Arcsec
    throughput_curve = 'wfos'
    rn = 2.5                            # Detector readnoise (e-)
    dark = 0.0                          # Detector dark-current (e-/s)

    # Temporary numbers that assume a given spectrograph PSF and LSF.
    # Assume 3 pixels per spectral and spatial FWHM.
    spatial_fwhm = 3.0
    spectral_fwhm = 3.0

    # Get source spectrum in 1e-17 erg/s/cm^2/angstrom. Currently, the
    # source spectrum is assumed to be
    #   - normalized by the total integral of the source flux 
    #   - independent of position within the source
    wave = get_wavelength_vector(args.wavelengths[0], args.wavelengths[1], args.wavelengths[2])
    emline_db = None if args.emline is None else read_emission_line_database(args.emline)
    spec = get_spectrum(wave, args.mag, emline_db=emline_db, redshift=args.redshift,
                        resolution=resolution)

    # Build the source surface brightness distribution with unity
    # integral; intrinsic is set to 1 for a point source
    intrinsic = 1. if args.sersic is None \
                    else source.OnSkySersic(1.0, args.sersic[0], args.sersic[1],
                                            ellipticity=args.sersic[2],
                                            position_angle=args.sersic[3], unity_integral=True)

    # Set the size of the map used to render the source
    size = args.fwhm*5 if args.sersic is None else max(args.fwhm*5, args.sersic[0]*5)
    sampling = args.fwhm/10 if args.sersic is None \
                    else min(args.fwhm/10, args.sersic[0]/10/args.sersic[1])

    # Check the rendering of the Sersic profile
    if args.sersic is not None:
        intrinsic.make_map(sampling=sampling, size=size)
        r, theta = intrinsic.semi.polar(intrinsic.X, intrinsic.Y)
        flux_ratio = 2 * numpy.sum(intrinsic.data[r < intrinsic.r_eff.value]) \
                        * numpy.square(sampling) / intrinsic.get_integral()
        if numpy.absolute(numpy.log10(flux_ratio)) > numpy.log10(1.05):
            warnings.warn('Difference in expected vs. map-rendered flux is larger than '
                          '5%: {0}%.'.format((flux_ratio-1)*100))

    # Construct the on-sky source distribution
    onsky = source.OnSkySource(args.fwhm, intrinsic, sampling=sampling, size=size)

    # Show the rendered source
#    pyplot.imshow(onsky.data, origin='lower', interpolation='nearest')
#    pyplot.show()

    # Get the sky spectrum
    sky_spectrum = spectrum.MaunakeaSkySpectrum()

    # Overplot the source and sky spectrum
#    ax = spec.plot()
#    ax = sky_spectrum.plot(ax=ax, show=True)

    # Get the atmospheric throughput
    atmospheric_throughput = efficiency.AtmosphericThroughput(airmass=args.airmass)

    # Set the telescope. Defines the aperture area and throughput
    # (nominally 3 aluminum reflections for Keck)
    telescope = telescopes.KeckTelescope()

    # Define the observing aperture; fiber diameter is in arcseconds,
    # center is 0,0 to put the fiber on the target center. "resolution"
    # sets the resolution of the fiber rendering; it has nothing to do
    # with spatial or spectral resolution of the instrument
    fiber = aperture.FiberAperture(0, 0, fiber_diameter, resolution=100)

    # Get the spectrograph throughput (circa June 2018; TODO: needs to
    # be updated). Includes fibers + foreoptics + FRD + spectrograph +
    # detector QE (not sure about ADC). Because this is the total
    # throughput, define a generic efficiency object.
    thru_db = numpy.genfromtxt(os.path.join(os.environ['ENYO_DIR'], 'data/efficiency',
                               'fiber_wfos_throughput.db'))
    spectrograph_throughput = efficiency.Efficiency(thru_db[:,3], wave=thru_db[:,0])

    # System efficiency combines the spectrograph and the telescope
    system_throughput = efficiency.SystemThroughput(wave=spec.wave,
                                                    spectrograph=spectrograph_throughput,
                                                    telescope=telescope.throughput)

    # Instantiate the detector; really just a container for the rn and
    # dark current for now. QE is included in fiber_wfos_throughput.db
    # file, so I set it to 1 here.
    det = detector.Detector(rn=rn, dark=dark, qe=1.0)

    # Extraction: makes simple assumptions about the detector PSF for
    # each fiber spectrum and mimics a "perfect" extraction, including
    # an assumption of no cross-talk between fibers. Ignore the
    # "spectral extraction".
    extraction = extract.Extraction(det, spatial_fwhm=spatial_fwhm, spatial_width=2*spatial_fwhm)

    # Perform the observation
    obs = Observation(telescope, sky_spectrum, fiber, args.time, det,
                      system_throughput=system_throughput,
                      atmospheric_throughput=atmospheric_throughput, airmass=args.airmass,
                      onsky_source_distribution=onsky, source_spectrum=spec, extraction=extraction,
                      per_resolution_element=not args.per_ang)

    # Construct the S/N spectrum
    snr = obs.snr(sky_sub=True)

    if args.ipython:
        embed()

    if args.plot:
        w,h = pyplot.figaspect(1)
        fig = pyplot.figure(figsize=(1.5*w,1.5*h))

        ax = fig.add_axes([0.1, 0.5, 0.8, 0.4])
        ax.set_xlim([wave[0], wave[-1]])
        ax.minorticks_on()
        ax.tick_params(which='major', length=8, direction='in', top=True, right=True)
        ax.tick_params(which='minor', length=4, direction='in', top=True, right=True)
        ax.grid(True, which='major', color='0.8', zorder=0, linestyle='-')
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.set_yscale('log')

        ax = spec.plot(ax=ax, label='Object')
        ax = sky_spectrum.plot(ax=ax, label='Sky')
        ax.legend()
        ax.text(-0.1, 0.5, r'Flux [10$^{-17}$ erg/s/cm$^2$/${\rm \AA}$]', ha='center', va='center',
                transform=ax.transAxes, rotation='vertical')
        
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.4])
        ax.set_xlim([wave[0], wave[-1]])
        ax.minorticks_on()
        ax.tick_params(which='major', length=8, direction='in', top=True, right=True)
        ax.tick_params(which='minor', length=4, direction='in', top=True, right=True)
        ax.grid(True, which='major', color='0.8', zorder=0, linestyle='-')

        ax = snr.plot(ax=ax)

        ax.text(0.5, -0.1, r'Wavelength [${\rm \AA}$]', ha='center', va='center',
                transform=ax.transAxes)
        ax.text(-0.1, 0.5, r'S/N per R element', ha='center', va='center',
                transform=ax.transAxes, rotation='vertical')

        pyplot.show()

    # Report
    g = efficiency.FilterResponse()
    r = efficiency.FilterResponse(band='r')
    print('-'*70)
    print('{0:^70}'.format('FOBOS S/N Calculation (v0.1)'))
    print('-'*70)
    print('Compute time: {0} seconds'.format(time.perf_counter() - t))
    print('Object g- and r-band AB magnitude: {0:.1f} {1:.1f}'.format(
                    spec.magnitude(g), spec.magnitude(r)))
    print('Sky g- and r-band AB surface brightness: {0:.1f} {1:.1f}'.format(
                    sky_spectrum.magnitude(g), sky_spectrum.magnitude(r)))
    print('Exposure time: {0:.1f} (s)'.format(args.time))
    print('Aperture Loss: {0:.1f}%'.format((1-obs.aperture_efficiency)*100))
    print('Extraction Loss: {0:.1f}%'.format((1-obs.extraction.spatial_efficiency)*100))
    print('Median S/N per resolution element: {0:.1f}'.format(numpy.median(snr.flux)))
    print('g-band weighted mean S/N per resolution element {0:.1f}'.format(
                numpy.sum(g(snr.wave)*snr.flux)/numpy.sum(g(snr.wave))))
    print('r-band weighted mean S/N per resolution element {0:.1f}'.format(
                numpy.sum(r(snr.wave)*snr.flux)/numpy.sum(r(snr.wave))))

