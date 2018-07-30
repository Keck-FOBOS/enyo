#!/bin/env/python3
# -*- encoding utf-8 -*-
"""
Define telesope parameters
"""

import numpy

class Telescope:
    """
    diameter is in m

    area in cm^2

    obstruction is in fraction of total area
    """
    def __init__(self, longitude, latitude, elevation, platescale, diameter=None, area=None,
                 obstruction=None):
        self.longitude = longitude
        self.latitude = latitude
        self.elevation = elevation
        self.platescale = platescale

        # If area is provided, use it directly:
        if area is not None:
            self.area = area
            if diameter is not None:
                raise ValueError('Cannot provide area and diameter; provide one or the other.')
            self.diameter = numpy.sqrt(area/numpy.pi)*2/100
            if obstruction is not None:
                warnings.warn('Obstruction and area provided, combining to get effective area.')
        elif diameter is not None:
            self.diameter = diameter
            self.area = numpy.pi*numpy.square(self.diameter*100/2)
        if obstruction is not None:
            self.area *= (1-obstruction)

class KeckTelescope(Telescope):
    def __init__(self):
        super(KeckTelescope, self).__init__(155.47833, 19.82833, 4160.0, 1.379, area=723674.)


class Observation:
    def __init__(self, airmass, sky_brightness, exposure_time, wavelength=None, band=None):
        if wavelength is None and band is None:
            raise ValueError('Must provide band or wavelength.')

        self.airmass = airmass
        self.sky_brightness = sky_brightness
        self.exposure_time = exposure_time
        self.wavelength = wavelength
        self.band = band

    @classmethod
    def from_date_telescope_target(cls, date, telescope, target, exposure_time, wavelength=None,
                                   band=None):
        raise NotImplementedError('Not yet implemented!')
        
#        # Use date to get lunar phase
#        lunar_cycle = 29.53 # days
#        day_since_new_moon = int(lunar_cycle*(0.5 
#                                    - astroplan.moon_phase_angle(Time(date)).value/2/numpy.pi))
#        sky_brightness = interpolate_sky_brightness(day_since_new_moon, wavelength=wavelength,
#                                                    band=band)
#        # Use date, telescope, and target to get airmass
#        targ = SkyCoord(target)
#        tele = EarthLocation(lat=telescope.latitude*units.deg, lon=telescope.longitude*units.deg,
#                             height=telescope.elevation*units.m)
#        targaltaz = targ.transform_to(AltAz(obstime=Time(date), location=tele))
#        airmass = targaltaz.secz
#
#        return cls(airmass, sky_brightness, exposure_time, wavelength=wavelength, band=band)



