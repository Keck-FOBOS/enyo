#!/usr/bin/env python3

import os
import time

import numpy

from matplotlib import pyplot

from enyo.etc import source, efficiency, telescopes, spectrum, extract, aperture
from enyo.etc.observe import Observation

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    t = time.clock()

    # Get the wavelength vector
    wave0 = 3100
    dlogw = 5e-5
    waven = 10000
    nwave = int((numpy.log10(waven)-numpy.log10(wave0))/dlogw + 1)
    wave = numpy.power(10., numpy.arange(nwave)*dlogw + numpy.log10(wave0))

    redshift = 0.0

    spectrum_type = 'emline'
    spectrum_type = 'blue'
    # Flux in 1e-17 erg/s/cm^2
    flux = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # Line names
    names = numpy.array(['Lya', '[OII]3727', '[OII]3729', 'Hb', '[OIII]4960', '[OIII]5008'])
    # Wavelength in angstroms
    restwave = numpy.array([ 1216., 3727.092, 3729.875, 4862.691, 4960.295, 5008.240])
    # Line FWHM in km/s
    fwhm = numpy.array([50., 50., 50., 50., 50., 50.])
    # Source spectrum will be in 1e-17 erg/s/cm^2/angstrom

    reff = 2.0
    sersic_n = 1.0
    ellipticity = 0.4
    position_angle = 45
    intrinsic = None          # If None, assumed to be point source

    seeing = 0.8
    airmass = 1.0

    fiber_diameter = 1.0

    throughput_curve = 'wfos'

#    # Set constant delta lambda such that the resolution is 3500 at 5000 angstroms
#    dispfwhm = 5000/3500
#    resolution = wave/dispfwhm
    # Set a constant FWHM sampling of the spectrum with a constant
    # resolution
    resolution = 3500

    # Detector stats
    pixelscale = fiber_diameter / 3     # Set 3 pixels per fiber diameter
    dispscale = dlogw                   # Set dispersion scale to match the spectrum
    rn = 3.                             # Detector readnoise (e-)
    dark = 0.                           # Detector dark-current (e-)
    qe = 0.9                            # Detector quantum efficiency (e-/photon)

    exposure_time = 3600

    # Approximate what the spot looks like on the detector
    spectral_fwhm = 1/resolution/dispscale      # Number of pixels per spectral FWHM
    spatial_fwhm = fiber_diameter/pixelscale    # Number of pixels per spatial FWHM
    
    # Set the number of pixels in each direction for the spectral
    # extraction
    spectral_width = 1                  # Keep single pixel extraction spectrally
    spatial_width = spatial_fwhm        # Extract 1 FWHM spatially

    # Get source spectrum.  In the current implementation, the source
    # spectrum is
    #   - expected to be normalized by the total integral of the source
    #     flux
    #   - expected to be independent of position within the source
    if spectrum_type == 'constant':
        source_spectrum = spectrum.Spectrum(wave, numpy.ones_like(wave, dtype=float), log=True)
    elif spectrum_type == 'blue':
        source_spectrum = spectrum.BlueGalaxySpectrum(redshift=redshift)
    elif spectrum_type == 'emline':
        source_spectrum = spectrum.EmissionLineSpectrum(wave, flux, restwave, fwhm, units='km/s',
                                                        redshift=redshift, resolution=resolution,
                                                        log=True)
#    source_spectrum.show()

    # Build the source surface brightness distribution with unity
    # integral
    if reff is not None and sersic_n is not None:
        intrinsic = source.OnSkySersic(1.0, reff, sersic_n, ellipticity=ellipticity,
                                       position_angle=position_angle, unity_integral=True)
    # Force a point source
    intrinsic = 1.
        
    # Construct the on-sky source distribution
    onsky = source.OnSkySource(seeing, intrinsic, sampling=0.1, size=20)

    # Get the sky spectrum
    sky_spectrum = spectrum.MaunakeaSkySpectrum()
#    sky_spectrum.show()

    # Get the atmospheric throughput
    atmospheric_throughput = efficiency.AtmosphericThroughput(airmass=airmass)

#    pyplot.plot(source_spectrum.wave, source_spectrum.flux)
#    pyplot.plot(sky_spectrum.wave, sky_spectrum.flux)
#    pyplot.plot(source_spectrum.wave, sky_spectrum.interp(source_spectrum.wave))
#    pyplot.show()

    # Telescope; mostly just used for aperture area for now
    telescope = telescopes.KeckTelescope()

    # Define the observing aperture; fiber diameter is in arcseconds,
    # center is 0,0 to put the fiber on the target center
    fiber = aperture.FiberAperture(0, 0, fiber_diameter, resolution=100)

    # Get the full system (top of telescope) throughput
    # - Spectrograph throughput
    if throughput_curve == 'wfos':
        data_file = os.path.join(os.environ['ENYO_DIR'], 'data/efficiency',
                                 'fiber_wfos_throughput.db')
        db = numpy.genfromtxt(data_file)
        spectrograph_throughput = efficiency.SpectrographThroughput(db[:,0], total=db[:,3])
    elif throughput_curve == 'desi':
        data_file = os.path.join(os.environ['ENYO_DIR'], 'data/efficiency',
                                 'desi_spectrograph_thru.txt')
        db = numpy.genfromtxt(data_file)
        spectrograph_throughput = efficiency.SpectrographThroughput(db[:,0], total=db[:,1])
    else:
        raise NotImplementedError('Spectrograph unknown: {0}'.format(throughput_curve))
    
    # Coating efficiency
    coating_efficiency = efficiency.Efficiency.from_file(os.path.join(os.environ['ENYO_DIR'],
                                                                      'data/efficiency',
                                                                      'aluminum.db'))
    
    # System efficiency including 3 aluminum bounces
    system_throughput = efficiency.SystemThroughput(source_spectrum.wave,
                                                    spectrograph=spectrograph_throughput,
                                                    surfaces=3, coating=coating_efficiency)

    # Instantiate the detector
    detector = efficiency.Detector(pixelscale=pixelscale, dispscale=dispscale, log=True,
                                        rn=rn, dark=dark, qe=qe)

    # Extraction
    extraction = extract.Extraction(detector, spatial_fwhm=spatial_fwhm,
                                    spatial_width=spatial_width, spectral_fwhm=spectral_fwhm,
                                    spectral_width=spectral_width)

    obs = Observation(onsky, source_spectrum, sky_spectrum, atmospheric_throughput, telescope,
                      fiber, system_throughput, detector, exposure_time, extraction)

    obs.simulate().show()

    obs.snr().show()

    exit()


    


    # Get the object and sky flux in electrons
    object_flux = source_spectrum.photon_flux() * self.exposure_time \
                                * self.source_spectrum.wave/resolution * self.telescope.area \
                                * self.atmospheric_extinction(self.source_spectrum.wave) \
                                * self.system_throughput(self.source_spectrum.wave)



    
    # Construct the spectrum


    s = SNRSpectrum(20., 1.0, 0.8, 1.0, 5000, 1., 0., 0.9)

    exit()

    reff = 0.1
    sersic_index = 1
    spectrum_type = 'constant'
    redshift = 0.0

    seeing = 0.8

    resolution = 5000       # lambda/delta lambda

    normalizing_band = 'g'  # Band used to normalize the source spectrum
    magnitude = 25          # AB magnitude of source
    exposure_time = 1       # Exposure time in seconds

    print('Elapsed time: {0} seconds'.format(time.clock() - t))



