#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Calculate the S/N for an observation.

Extraction:
    - fiber-extraction aperture (pixels, optimal?)
    - fiber PSF FWHM on detector (pixels)

    - spectral resolution (R)
    - dispersion on detector (A per pixel)

Detector:
    - arcsec / pixel (spatial)
    - angstrom / pixel (spectral)
    - detector readnoise (e-)
    - detector darkcurrent (e-/hour)

Throughput
    - spectrograph throughput
        - detector QE
        - camera efficiency
        - grating efficiency
        - focal-plane to grating tansmission efficiency
            - foreoptics, lenslet, coupling, fiber transmission, FRD,
              collimator, dichroic(s)

    - top-of-telescope throughput
        - telescope efficiency
        - spectrograph efficiency

Point-source Aperture losses
    - fiber diameter (arcsec, mm)
    - focal-plane plate scale (arcsec/mm)
    - seeing (arcsec)
    - pointing error (arcsec)

*Source:
    - source spectrum (1e-17 erg/s/cm^2/angstrom)

    - point source
        - source mag (at wavelength/in band above)

    - extended source
        - source surface brightness (at wavelength/in band above)
        - source size (kpc/arcsec)
        - cosmology

    - source velocity/redshift

*Sky:
    - sky spectrum (1e-17 erg/s/cm^2/angstrom)

Observation:
    - obs date
    - lunar phase/illumination (days, fraction)
    - sky coordinates
    - wavelength for calculation (angstroms)
    - band for calculation
    - sky surface brightness (mag/arcsec^2, at wavelength/in band above)
    - airmass
    - exposure time (s)

Telescope:
    - location
    - telescope diameter (or effective) (m^2)
    - central obstruction

"""

import os
import numpy

from matplotlib import pyplot

from . import onskysource, efficiency, telescopes, spectrum, extract

class SNRSpectrum:
    def __init__(self, magnitude, exposure_time, seeing, airmass, resolution, rn, dark, qe,
                 spectrograph='wfos', fiber_diameter=1., spectral_fwhm=3, spectral_width=1,
                 spatial_fwhm=3, spatial_width=1, surface_brightness=None, reff=None,
                 sersic_index=None, redshift=0.0, spectrum_type='constant', normalizing_band='g',
                 sky='maunakea'):
        
        self.redshift = redshift                # Object redshift
        self.magnitude = magnitude              # AB magnitude of source
        self.surface_brightness = surface_brightness    # Surface brightness in AB mag / arcsec^2
        self.normalizing_band = normalizing_band        # Magnitude/surface brightness band
        self.exposure_time = exposure_time      # Exposure time in seconds
        self.seeing = seeing                    # FWHM of on-sky PSF

        self.airmass = airmass                  # Airmass of the observation
                                                # Allow to get this from
                                                # telescope, date, and
                                                # target

        self.spectrograph = spectrograph        # Name of the spectrograph
        self.fiber_diameter = fiber_diameter    # Size of the fiber in arcsec
        self.resolution = resolution            # lambda/delta lambda of spectrograph
        self.spectral_fwhm = spectral_fwhm      # FWHM of resolution element in pixels
        self.spectral_width = spectral_width    # Number of FWHM for spectral extraction
        self.spatial_fwhm = spatial_fwhm        # FWHM of fiber PSF in pixels
        self.spatial_width = spatial_width      # Number of FWHM for spatial extraction

        # Instantiate the detector
        # TODO: Read the QE from a file
        self.detector = efficiency.Detector(rn=rn,      # Detector readnoise (e-)
                                            dark=dark,  # Detector dark-current (e-/s)
                                            qe=qe)      # Detector quantum efficiency

        # Instantiate telescope
        print('telescope')
        self.telescope = telescopes.KeckTelescope()

        # Get sky
        print('sky')
        if sky == 'maunakea':
            self.sky_spectrum = spectrum.MaunakeaSkySpectrum()
        else:
            raise NotImplementedError('Sky spectrum from {0} is not implemented.'.format(sky))
        # Rescale for moon phase given telescope, date, and target

        # Source spectrum
        print('source')
        if spectrum_type == 'constant':
            self.source_spectrum = spectrum.Spectrum(self.sky_spectrum.wave,
                                                     numpy.ones_like(self.sky_spectrum.wave,
                                                                     dtype=float),
                                                     log=True)
        else:
            raise NotImplementedError('Souce spectrum unknown: {0}'.format(spectrum_type))
        self.source_spectrum = spectrum.BlueGalaxySpectrum()

        # Rescale source spectrum to the input magnitude or surface
        # brightness
        print('band')
        self.band = spectrum.FilterResponse(band=self.normalizing_band)
        print('source rescale')
        self.source_spectrum.rescale_magnitude(self.band, self.magnitude 
                                if self.surface_brightness is None else self.surface_brightness)

        # Spectrograph throughput
        print('spectrograph')
        if self.spectrograph == 'wfos':
            data_file = os.path.join(os.environ['ENYO_DIR'], 'data/efficiency',
                                     'fiber_wfos_throughput.db')
            db = numpy.genfromtxt(data_file)
            self.spectrograph_throughput = efficiency.SpectrographThroughput(db[:,0],
                                                                             total=db[:,3])
        elif self.spectrograph == 'desi':
            data_file = os.path.join(os.environ['ENYO_DIR'], 'data/efficiency',
                                     'desi_spectrograph_thru.txt')
            db = numpy.genfromtxt(data_file)
            self.spectrograph_throughput = efficiency.SpectrographThroughput(db[:,0],
                                                                             total=db[:,1])
        else:
            raise NotImplementedError('Spectrograph unknown: {0}'.format(self.spectrograph))


        # System efficiency including 3 aluminum bounces
        print('system')
        self.coating_efficiency = efficiency.Efficiency.from_file(
                    os.path.join(os.environ['ENYO_DIR'], 'data/efficiency', 'aluminum.db'))
        self.system_throughput = efficiency.SystemThroughput(self.source_spectrum.wave,
                                                    spectrograph=self.spectrograph_throughput,
                                                             surfaces=3,
                                                             coating=self.coating_efficiency)

        # Atmosphere
        print('atmosphere')
        self.atmospheric_extinction = efficiency.AtmosphericThroughput(airmass=self.airmass)

        # Get the object and sky flux in electrons
        print('object flux')
        self.object_flux = self.source_spectrum.photon_flux() * self.exposure_time \
                                * self.source_spectrum.wave/resolution * self.telescope.area \
                                * self.atmospheric_extinction(self.source_spectrum.wave) \
                                * self.system_throughput(self.source_spectrum.wave)

        # Apply aperture limits if source flux based on total magnitude,
        # or integrate the source within the fiber area if given as a
        # uniform surface brightness
        if self.surface_brightness is None:
            # Intrinsic on-sky source distribution:
            #   - Point source if reff and sersic_index are not provided
            #   - Sersic profile otherwise
            # It doesn't matter what the integral of source is,
            # aperture_efficiency normalizes it out
            self.source = 1.0 if reff is None or sersic_index is None \
                        else onskysource.OnSkySersic(1.0, self.reff, self.sersic_index,
                                                     unity_integral=True)
            # Aperture Efficiency
            print('aperture efficiency')
            self.aperture_efficiency = efficiency.ApertureEfficiency(self.source_spectrum.wave,
                                                                     self.fiber_diameter,
                                                                     source=self.source,
                                                                     seeing=self.seeing)
        else:
            self.source = None
            self.aperture_efficiency = None

        self.object_flux *= (numpy.pi * numpy.square(self.fiber_diameter/2)
                                if self.aperture_efficiency is None else
                                    self.aperture_efficiency(self.source_spectrum.wave))

        # Total sky flux; sky is always assumed to be uniform over the
        # aperture
        print('sky flux')
        self.sky_flux = self.sky_spectrum.photon_flux() * self.exposure_time \
                                * self.source_spectrum.wave/resolution * self.telescope.area \
                                * self.system_throughput(self.source_spectrum.wave) \
                                * numpy.pi * numpy.square(self.fiber_diameter/2)


        # Extraction
        self.extraction = extract.Extraction(spatial_fwhm=self.spatial_fwhm,
                                             spatial_width=self.spatial_width,
                                             spectral_fwhm=self.spectral_fwhm,
                                             spectral_width=self.spectral_width)
        
        # Get S/N from a basic sum extraction
        self.snr = self.extraction.sum_snr(self.object_flux, self.sky_flux, self.detector.rn,
                                           self.exposure_time*self.detector.dark)

    def reset_detector(self, rn, dark):
        """Reset the detector stats and recalculate.
        .. warning::
            DOES NOT INCLUDE QE CHANGES
        """
        self.detector = Detector(rn=rn, dark=dark)
        self.snr = extraction.sum_snr(self.object_flux, self.sky_flux, self.detector.rn,
                                      self.exposure_time*self.detector.dark)
    
