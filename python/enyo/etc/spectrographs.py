#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Detector class
"""

import os
import numpy

from . import telescopes, observe, kernel, detector, efficiency, optical

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
                 throughput, opticalmodel, scramble=False):
        self.platescale = platescale            # focal-plane plate scale in mm/arcsec
        self.cwl = cwl                          # central wavelength in angstroms
        self.dispersion = dispersion            # linear dispersion in A/mm
        self.spectral_range = spectral_range    # free spectral range (Delta lambda)
        self.detector = arm_detector            # Detector instance
        self.kernel = arm_kernel                # Monochromatic kernel
        self.throughput = throughput            # Should describe the throughput from the focal
                                                # plane to the detector, including detector QE;
                                                # see `SpectrographThroughput`
        self.opticalmodel = opticalmodel        # Optical model used to propagate rays from the
                                                # focal plane to the camera detector
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

    def twod_spectrum(self, sky_spectrum, spec_aperture, source_distribution=None,
                      source_spectrum=None, wave_lim=None, field_coo=None, rectilinear=False):
        optical_args = {} if rectilinear \
                            else {'field_coo': field_coo, 'opticalmodel':self.opticalmodel}
        return observe.twod_spectrum(sky_spectrum, spec_aperture, self.kernel, self.platescale,
                                     self.dispersion, self.detector.pixelsize,
                                     source_distribution=source_distribution,
                                     source_spectrum=source_spectrum, thresh=1e-10,
                                     wave_lim=wave_lim, **optical_args)

    def observe(sky, sky_spectrum, spec_aperture, exposure_time, airmass, onsky_source=None,
                source_spectrum=None, extraction=None):
        """
        Take an observation through an aperture.
        """
        pass


class TMTWFOSBlue(SpectrographArm):
    def __init__(self, setting='lowres'):
        if setting not in TMTWFOSBlue.valid_settings():
            raise ValueError('Setting {0} not known.'.format(setting))

        # Setup the telescope
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

        opticalmodel = TMTWFOSBlueOpticalModel(setting=setting)

        if setting == 'lowres':
            #   Central wavelength is 4350 angstroms
            #   Linear dispersion is 13.3 angstroms per mm
            #   Free spectral range is 2500 angstroms
            super(TMTWFOSBlue, self).__init__(platescale, 4350., 13.3, 2500., arm_detector,
                                              arm_kernel, throughput, opticalmodel)

    @staticmethod
    def valid_settings():
        return ['lowres']


class TMTWFOSRed(SpectrographArm):
    def __init__(self, setting='lowres'):
        if setting not in TMTWFOSRed.valid_settings():
            raise ValueError('Setting {0} not known.'.format(setting))

        # Setup the telescope
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

        opticalmodel = TMTWFOSRedOpticalModel(setting=setting)

        if setting == 'lowres':
            #   Central wavelength is 7750 angstroms
            #   Linear dispersion is 23.7 angstroms per mm
            #   Free spectral range is 4500 angstroms
            super(TMTWFOSRed, self).__init__(platescale, 7750., 23.7, 4500., arm_detector,
                                             arm_kernel, throughput, opticalmodel)

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
        self.arms = {'blue': TMTWFOSBlue(setting=setting), 'red': TMTWFOSRed(setting=setting)}

    def monochromatic_image(self, sky, spec_aperture, onsky_source=None, arm=None):
        """
        Generate monochromatic images of the source through the
        aperture in one or more of the spectrograph arms.
        """
        if arm is not None:
            return self.arms[arm].monochromatic_image(sky, spec_aperture, onsky_source=onsky_source)
        return dict([(key, a.monochromatic_image(sky, spec_aperture, onsky_source=onsky_source))
                         for key,a in self.arms.items()])

    def twod_spectrum(self, sky_spectrum, spec_aperture, source_distribution=None,
                      source_spectrum=None, wave_lim=None, arm=None, field_coo=None,
                      rectilinear=False):
        """
        Generate a 2D spectrum of the source through the aperture in
        one or more of the spectrograph arms.
        """
        if arm is not None:
            return self.arms[arm].twod_spectrum(sky_spectrum, spec_aperture,
                                                source_distribution=source_distribution,
                                                source_spectrum=source_spectrum, wave_lim=wave_lim,
                                                field_coo=field_coo, rectilinear=rectilinear)
        return dict([(key, a.twod_spectrum(sky_spectrum, spec_aperture,
                                           source_distribution=source_distribution,
                                            source_spectrum=source_spectrum, wave_lim=wave_lim,
                                            field_coo=field_coo, rectilinear=rectilinear))
                         for key,a in self.arms.items()])

#    def observe(self, source_distribution, source_spectrum, sky_distribution, sky_spectrum,
#                spec_aperture, airmass, exposure_time, extraction):
#        """
#        Returns the total spectrum and variance, sky spectrum
#        """
#        # TODO: sky_spectrum should default to Maunakea


class TMTWFOSBlueOpticalModel(optical.OpticalModelInterpolator):
    def __init__(self, setting='lowres'):
        if setting == 'lowres':
            modelfile = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos',
                                     'Blue_Low_Spot_Data_2020_150.txt')
        else:
            raise NotImplementedError('Setting {0} not yet recognized.'.format(setting))
        xf, yf, wave, xc, yc, rays = numpy.genfromtxt(modelfile).T
        indx = rays > 0
        # Convert field coordinates from arcmin to arcsec, wavelengths
        # from micron to angstroms, and percentage of incident rays to
        # a fraction
        super(TMTWFOSBlueOpticalModel, self).__init__(60*xf[indx], 60*yf[indx], 10000*wave[indx],
                                                      xc[indx], yc[indx], vignette=0.01*rays[indx])


class TMTWFOSRedOpticalModel(optical.OpticalModelInterpolator):
    def __init__(self, setting='lowres'):
        if setting == 'lowres':
            modelfile = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos',
                                     'Red_Low_Spot_Data_2020_150.txt')
        else:
            raise NotImplementedError('Setting {0} not yet recognized.'.format(setting))
        xf, yf, wave, xc, yc, rays = numpy.genfromtxt(modelfile).T
        indx = rays > 0
        # Convert field coordinates from arcmin to arcsec, wavelengths
        # from micron to angstroms, and percentage of incident rays to
        # a fraction
        super(TMTWFOSRedOpticalModel, self).__init__(60*xf[indx], 60*yf[indx], 10000*wave[indx],
                                                     xc[indx], yc[indx], vignette=0.01*rays[indx])



