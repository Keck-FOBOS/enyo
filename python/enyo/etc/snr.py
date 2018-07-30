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

from . import onskysource

class SNRSpectrum:
    def __init__(magnitude, exposure_time, seeing, airmass, resolution, rn, dark, qe,
                 spectrograph='wfos', spectral_fwhm=3, spectral_width=1, spatial_fwhm=3,
                 spatial_width=1, surface_brightness=None, reff=None, sersic_index=None,
                 redshift=0.0, spectrum_type='constant', normalizing_band='g', sky='maunakea'):
        
        self.redshift = redshift                # Object redshift
        self.magnitude = magnitude              # AB magnitude of source
        self.surface_brightness = surface_brightness    # Surface brightness in AB mag / arcsec^2
        self.exposure_time = exposure_time      # Exposure time in seconds
        self.seeing = seeing                    # FWHM of on-sky PSF

        self.spectrograph = spectrograph        # Name of the spectrograph
        self.resolution = resolution            # lambda/delta lambda of spectrograph
        self.spectral_fwhm = spectral_fwhm      # FWHM of resolution element in pixels
        self.spectral_width = spectral_width    # Number of FWHM for spectral extraction
        self.spatial_fwhm = spatial_fwhm        # FWHM of fiber PSF in pixels
        self.spatial_width = spatial_width      # Number of FWHM for spatial extraction

        # Instantiate the detector
        # TODO: Read the QE from a file
        self.detector = Detector(rn=rn,         # Detector readnoise (e-)
                                 dark=dark,     # Detector dark-current (e-/s)
                                 qe=qe)         # Detector quantum efficiency

        # Instantiate telescope
        self.telescope = KeckTelescope()

        # Get sky
        if sky == 'maunakea':
            self.sky_spectrum = MaunakeaSkySpectrum()
        else:
            raise NotImplementedError('Sky spectrum from {0} is not implemented.'.format(sky))
        # Rescale for moon phase given telescope, date, and target

        # Intrinsic on-sky source distribution: It doesn't matter what
        # the integral of source is, aperture_efficiency normalizes it
        # out
        self.source = None if reff is None or sersic_index is None \
                        else OnSkySersic(1.0, self.reff, self.sersic_index, unity_integral=True)

        # Source spectrum
        if spectrum_type == 'constant':
            self.source_spectrum = Spectrum(self.sky_spectrum.wave,
                                            numpy.ones_like(self.sky_spectrum.wave, dtype=float),
                                            log=True)
        else:
            raise NotImplementedError('Souce spectrum unknown: {0}'.format(spectrum_type))

        # Rescale source spectrum to the input magnitude or surface
        # brightness
        self.band = FilterResponse(band=self.normalizing_band)
        self.source_spectrum.rescale_magnitude(self.band, self.magnitude 
                                if self.surface_brightness is None else self.surface_brightness)

        # Spectrograph throughput
        if self.spectrograph == 'wfos':
            data_file = os.path.join(os.environ['ENYO_DIR'], 'data/efficiency',
                                     'fiber_wfos_throughput.db')
            db = numpy.genfromtxt(data_file)
            self.spectrograph_throughput = SpectrographThroughput.from_total(db[:,0], db[:,3])
        elif self.spectrograph == 'desi':
            data_file = os.path.join(os.environ['ENYO_DIR'], 'data/efficiency',
                                     'desi_spectrograph_thru.txt')
            db = numpy.genfromtxt(data_file)
            self.spectrograph_throughput = SpectrographThroughput.from_total(db[:,0], db[:,1])
        else:
            raise NotImplementedError('Spectrograph unknown: {0}'.format(self.spectrograph))

        # System efficiency including 3 aluminum bounces
        self.coating_efficiency = Efficiency.from_file(os.path.join(os.environ['ENYO_DIR'],
                                                       'data/efficiency', 'aluminum.db'))
        self.system_throughput = SystemThroughput(self.source_spectrum.wave,
                                                  spectrograph=self.spectrograph_throughput,
                                                  surfaces=3, coating=self.coating_efficiency)

        # Aperture Efficiency
        self.aperture_efficiency = ApertureEfficiency(self.source_spectrum.wave,
                                                      self.fiber_diameter, source=self.source,
                                                      seeing=self.seeing)

        # Get the object and sky flux in electrons
        self.object_flux = self.source_spectrum.photon_flux() * self.telescope.area \
                                * self.exposure_time \
                                * self.aperture_efficiency.interp(self.source_spectrum.wave)
        self.sky_flux = self.sky_spectrum.photon_flux() * self.telescope.area \
                                * self.exposure_time \
                                * numpy.pi * numpy.square(self.fiber_diameter/2)

        # Extraction
        self.extraction = Extraction(spatial_fwhm=self.spatial_fwhm,
                                     spatial_width=self.spatial_width,
                                     spectral_fwhm=self.spectral_fwhm,
                                     spectral_width=self.spectral_width)
        
        # Get S/N from extraction extraction
        self.snr = extraction.sum_snr(self.object_flux, self.sky_flux, self.detector.rn,
                                      self.exposure_time*self.detector.dark)

    def reset_detector(self, rn, dark):
        """Reset the detector stats and recalculate.
        .. warning::
            DOES NOT INCLUDE QE CHANGES
        """
        self.detector = Detector(rn=rn, dark=dark)
        self.snr = extraction.sum_snr(self.object_flux, self.sky_flux, self.detector.rn,
                                      self.exposure_time*self.detector.dark)
    
