#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Detector class
"""

import os
import numpy

from . import telescopes, observe, kernel, detector, efficiency

#class Spectrograph:
#    """
#    Base class.
#    """
#    def __init__(self):
#        pass
#
#class MultiArmSpectrograph:
#    """
#    Base class.
#    """
#    def __init__(self):
#        pass

class SpectrographArm:
    """
    Base class for a spectrograph arm.
    """
    def __init__(self, platescale, cwl, dispersion, spectral_range, arm_detector, arm_kernel,
                 throughput, scramble=False):
        self.platescale = platescale            # focal-plane plate scale in mm/arcsec
        self.cwl = cwl                          # central wavelength in angstroms
        self.dispersion = dispersion            # linear dispersion in A/mm
        self.spectral_range = spectral_range    # free spectral range (Delta lambda)
        self.detector = arm_detector            # Detector instance
        self.kernel = arm_kernel                # Monochromatic kernel
        self.throughput = throughput            # Should describe the throughput from the focal
                                                # plane to the detector, including detector QE;
                                                # see `SpectrographThroughput`
        self.scramble = scramble                # Does the source image get scrambled by the
                                                # entrance aperture?

    @property
    def pixelscale(self):
        return self.detector.pixelsize/self.platescale     # arcsec/pixel

    @property
    def dispscale(self):
        return self.detector.pixelsize*self.dispersion      # A/pixel

    def monochromatic_image(self, sky, spec_aperture, onsky_source=None):
        """
        Construct a monochromatic image of a source through an
        aperture as observed by this spectrograph arm.
        """
        return observe.monochromatic_image(sky, spec_aperture, self.kernel, self.platescale,
                                           self.detector.pixelsize, onsky_source=onsky_source,
                                           scramble=self.scramble)

    def observe(sky, sky_spectrum, spec_aperture, exposure_time, airmass, onsky_source=None,
                source_spectrum=None, extraction=None):
        """
        Take an observation through an aperture.
        """
        pass


class TMTWFOSBlue(SpectrographArm):
    def __init__(self, setting='lowres', telescope=None):
        if setting not in TMTWFOSBlue.valid_settings():
            raise ValueError('Setting {0} not known.'.format(setting))

        if telescope is None:
            telescope = telescopes.TMTTelescope()
        # Plate-scale in mm/arcsec assuming the camera fratio is 2
        platescale = telescope.platescale * 2 / telescope.fratio
        # Assume camera yields 0.2 arcsec FWHM in both dimensions
        spatial_FWHM, spectral_FWHM = numpy.array([0.2, 0.2])*platescale
        # Assign the kernel without setting the pixel sampling
        arm_kernel = kernel.SpectrographGaussianKernel(spatial_FWHM, spectral_FWHM)
        # The detector
        qe_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'efficiency', 'detectors',
                               'itl_sta_blue.db')
        arm_detector = detector.Detector((4*4096, 2*4096), pixelsize=0.015, rn=2.5, dark=0.1,
                                         qe=efficiency.Efficiency.from_file(qe_file))
        # Focal-plane up to, but not including, detector throughput
        throughput_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'efficiency', 
                                       'wfos_throughput.db')
        pre_detector_eta = efficiency.Efficiency.from_file(throughput_file)
        # Total throughput
        throughput = efficiency.SpectrographThroughput(detector=arm_detector,
                                                       other=pre_detector_eta)

        if setting == 'lowres':
            #   Central wavelength is 4350 angstroms
            #   Linear dispersion is 13.3 angstroms per mm
            #   Free spectral range is 2500 angstroms
            super(TMTWFOSBlue, self).__init__(platescale, 4350., 13.3, 2500., arm_detector,
                                              arm_kernel, throughput)

    @staticmethod
    def valid_settings():
        return ['lowres']


class TMTWFOSRed(SpectrographArm):
    def __init__(self, setting='lowres', telescope=None):
        if setting not in TMTWFOSRed.valid_settings():
            raise ValueError('Setting {0} not known.'.format(setting))

        if telescope is None:
            telescope = telescopes.TMTTelescope()
        # Plate-scale in mm/arcsec assuming the camera fratio is 2
        platescale = telescope.platescale * 2 / telescope.fratio
        # Assume camera yields 0.2 arcsec FWHM in both dimensions
        spatial_FWHM, spectral_FWHM = numpy.array([0.2, 0.2])*platescale
        # Assign the kernel without setting the pixel sampling
        arm_kernel = kernel.SpectrographGaussianKernel(spatial_FWHM, spectral_FWHM)
        # The detector
        qe_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'efficiency', 'detectors',
                               'itl_sta_red.db')
        arm_detector = detector.Detector((4*4096, 2*4096), pixelsize=0.015, rn=2.5, dark=0.1,
                                         qe=efficiency.Efficiency.from_file(qe_file))
        # Focal-plane up to, but not including, detector throughput
        throughput_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'efficiency', 
                                       'wfos_throughput.db')
        pre_detector_eta = efficiency.Efficiency.from_file(throughput_file)
        # Total throughput
        throughput = efficiency.SpectrographThroughput(detector=arm_detector,
                                                       other=pre_detector_eta)

        if setting == 'lowres':
            #   Central wavelength is 7750 angstroms
            #   Linear dispersion is 23.7 angstroms per mm
            #   Free spectral range is 4500 angstroms
            super(TMTWFOSRed, self).__init__(platescale, 7750., 23.7, 4500., arm_detector,
                                             arm_kernel, throughput)

    @staticmethod
    def valid_settings():
        return ['lowres']

class TMTWFOS:  #(MultiArmSpectrograph)
    """
    Instantiate a setting of the WFOS spectrograph on TMT.
    """
    # TODO: Sky stuff should default to Maunakea and be defined by target position and moon-phase...
    # TODO: Allow airmass to be defined by target position and UT start of observation
    def __init__(self, setting='lowres'):
        self.telescope = telescopes.TMTTelescope()
        # TODO: Allow settings to be different for each arm.
        self.arms = {'blue': TMTWFOSBlue(setting=setting, telescope=self.telescope),
                      'red': TMTWFOSRed(setting=setting, telescope=self.telescope)}

    def monochromatic_image(self, sky, spec_aperture, onsky_source=None, arm=None):
        """
        Generate monochromatic images of the source through the
        aperture in one or more of the spectrograph arms.
        """
        if arm is not None:
            return self.arms[arm].monochromatic_image(sky, spec_aperture, onsky_source=onsky_source)
        return dict([(key, a.monochromatic_image(sky, spec_aperture, onsky_source=onsky_source))
                         for key,a in self.arms.items()])

#    def observe(self, source_distribution, source_spectrum, sky_distribution, sky_spectrum,
#                spec_aperture, airmass, exposure_time, extraction):
#        """
#        Returns the total spectrum and variance, sky spectrum
#        """
#        # TODO: sky_spectrum should default to Maunakea





