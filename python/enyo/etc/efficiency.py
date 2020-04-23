"""
Various efficiency calculations
"""
import os
import warnings
import inspect

from IPython import embed

import numpy

from matplotlib import pyplot

from scipy import interpolate

from astropy import units

class Efficiency:
    """
    Base class for efficiency data.

    Provided a wavelength independent efficiency value (`eta`), or a
    sampled vector of efficiency (`eta`) vs wavelength (`wave`), this
    class mainly just allows for access and interpolation of those
    data. When provided as a function of wavelength, the efficiency
    is assumed to be 0 outside the bounds of the provided wavelength
    vector.

    Args:
        eta (:obj:`float`, array-like):
            Constant or 1D efficiency data.
        wave (array-like, optional):
            1D array with wavelengths in angstroms.

    Attributes:
        interpolator (`scipy.interpolate.interp1d`):
            Linear interpolator used to sample efficiency at any
            wavelength, if vectors are provided.
    """
    def __init__(self, eta, wave=None):
        if wave is None:
            # eta has to be a constant
            if hasattr(eta, '__len__'):
                raise TypeError('If instantiated without a wavelength vector, the efficiency '
                                'must be a wavelength-independent contant.')
            self.interpolator = None
            self._eta = eta

        else:
            _wave = numpy.atleast_1d(wave)
            _eta = numpy.atleast_1d(eta)
            if _eta.ndim > 1 or len(_eta) == 1:
                raise ValueError('When providing wavelengths, efficiency must be a 1D vector '
                                 'with more than one element.')
            if _wave.shape != _eta.shape:
                raise ValueError('Efficiency and wavelengths must have the same shape.')
            self.interpolator = interpolate.interp1d(_wave, _eta, assume_sorted=True,
                                                     bounds_error=False, fill_value=0.0)
            self._eta = None
    
    @classmethod
    def from_file(cls, data_file, wave_units='angstrom'):
        """
        Read from an ascii file
        """
        if not os.path.isfile(data_file):
            raise FileNotFoundError('File does not exist: {0}'.format(data_file))
        db = numpy.genfromtxt(data_file)
        return cls(db[:,1], wave=db[:,0]*units.Unit(wave_units).to('angstrom'))

    def __call__(self, wave):
        _wave = numpy.atleast_1d(wave)
        if self.interpolator is None:
            return self._eta if _wave.size == 1 \
                        else numpy.full(_wave.shape, self._eta, dtype=float)
        _eta = self.interpolator(_wave)
        return _eta if hasattr(wave, '__len__') else _eta[0]

    def __getitem__(self, k):
        if self.interpolator is None:
            # TODO: Handle if k is a slice...
            warnings.warn('Efficiency is not a vector!  Returning constant value.')
            return self._eta
        return self.interpolator.y[k]

    @property
    def wave(self):
        if self.interpolator is None:
            warnings.warn('Efficiency is wavelength independent.')
            return None
        return self.interpolator.x

    @property
    def eta(self):
        if self.interpolator is None:
            return self._eta
        return self.interpolator.y

    def rescale(self, scale):
        """
        Scale must either be a single value or match the size of the
        existing eta vector.
        """
        if self.interpolator is None:
            self._eta *= scale
        else:
            self.interpolator.y *= scale


class CombinedEfficiency(Efficiency):
    """
    A class that combines multiple efficiencies that can be accessed
    separately or act as a single efficiency.

    Args:
        efficiencies (:obj:`list`, :obj:`dict`):
            The set of efficiencies to combine. Nominally this should
            be a dictionary that gives the efficiencies and a keyword
            identifier for each. A list can be entered, meaning that
            the efficiencies can only be access by their index, not a
            keyword.
        wave (array-like):
            Wavelengths of/for efficiency measurements.

    Attributes:
        efficiencies (:obj:`dict`):
            The efficiencies combined. Access to individual
            efficiencies is by keyword; if keywords not provided,
            access is by single integer index.
    """
    def __init__(self, efficiencies, wave=None):
        if isinstance(efficiencies, list):
            self.efficiencies = dict([(i,eff) for i,eff in enumerate(efficiencies)])
        elif isinstance(efficiencies, dict):
            self.efficiencies = efficiencies
        else:
            raise TypeError('Efficiencies to include must provided as a list or dict.')

        # Make sure the components of self are Efficiency objects
        for eff in self.efficiencies.values():
            if not isinstance(eff, Efficiency):
                raise TypeError('Each element of input must be an Efficiency object.')

        if wave is None:
            # Consolidate wavelengths from other efficiencies
            wave = numpy.empty(0, dtype=float)
            for inp in self.efficiencies.values():
                if inp.wave is None:
                    continue
                wave = numpy.append(wave, inp.wave)
            wave = None if len(wave) == 0 else numpy.sort(wave)
            if wave is None:
                warnings.warn('No wavelengths provided for any efficiencies to combine.')

        # Construct the total efficiency
        total = 1. if wave is None else numpy.ones_like(wave, dtype=float)
        for eff in self.efficiencies.values():
            total *= (eff.eta if wave is None else eff(wave))

        super(CombinedEfficiency, self).__init__(total, wave=wave)

    @classmethod
    def from_total(cls, total, wave=None):
        return cls({'total': Efficiency(total, wave=wave)})

    def keys(self):
        return self.efficiencies.keys()

    def __getitem__(self, key):
        """Return the specified efficiency."""
        return self.efficiencies[key]


class FiberThroughput(Efficiency):
    def __init__(self, fiber='polymicro'):
        data_file = FiberThroughput.select_data_file(fiber)
        if not os.path.isfile(data_file):
            raise FileNotFoundError('No file: {0}'.format(data_file))
        db = numpy.genfromtxt(data_file)
        super(FiberThroughput, self).__init__(db[:,1], wave=db[:,0])

    @staticmethod
    def select_data_file(fiber):
        if fiber == 'polymicro':
            return os.path.join(os.environ['ENYO_DIR'], 'data', 'efficiency', 'fibers',
                                'polymicro.db')
        raise NotImplementedError('Unknown fiber type: {0}'.format(fiber))


class FilterResponse(Efficiency):
    """
    The efficiency of a broad-band filter.

    Args:
        band (:obj:`str`, optional):
            The band to use.  Options are for the response functions in
            the data/broadband_filters directory.

    Raises:
        FileNotFoundError:
            Raised if the default file for the given band is not
            available.
    """
    def __init__(self, band='g'):
        data_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'broadband_filters',
                                 'gunn_2001_{0}_response.db'.format(band))
        if not os.path.isfile(data_file):
            raise FileNotFoundError('No file: {0}'.format(data_file))
        db = numpy.genfromtxt(data_file)
        super(FilterResponse, self).__init__(db[:,1], wave=db[:,0])


class AtmosphericThroughput(Efficiency):
    def __init__(self, airmass=1.0, location='maunakea'):
        if location == 'maunakea':
            db = numpy.genfromtxt(os.path.join(os.environ['ENYO_DIR'], 'data', 'sky',
                                  'mauna_kea_extinction.db'))
        else:
            raise NotImplementedError('Extinction unknown at {0}.'.format(location))
        self.airmass = airmass
        super(AtmosphericThroughput, self).__init__(numpy.power(10, -0.4*db[:,1]*self.airmass),
                                                    wave=db[:,0])

    def reset_airmass(self, airmass):
        self.rescale(numpy.power(self.interpolator.y, (airmass/self.airmass -1)))


class SpectrographThroughput(CombinedEfficiency):
    """
    Define the system throughput from the telescope focal plane to the detector.
    """
    def __init__(self, wave=None, coupling=None, fibers=None, grating=None, camera=None,
                 detector=None, other=None):
        values = inspect.getargvalues(inspect.currentframe())
        keys = numpy.array(values.args[2:])
        objects = numpy.array([values.locals[key] for key in keys])
        indx = numpy.array([o is not None for o in objects])
        efficiencies = dict([(k,o) for k,o in zip(keys[indx], objects[indx])])
        super(SpectrographThroughput, self).__init__(efficiencies, wave=wave)


class SystemThroughput(CombinedEfficiency):
    """
    Define the system throughput from the top of the telescope to the detector.
    """
    # TODO: Also allow for 'other' here?
    def __init__(self, wave=None, spectrograph=None, telescope=None):
        values = inspect.getargvalues(inspect.currentframe())
        keys = numpy.array(values.args[2:])
        objects = numpy.array([values.locals[key] for key in keys])
        indx = numpy.array([o is not None for o in objects])
        efficiencies = dict([(k,o) for k,o in zip(keys[indx], objects[indx])])
        super(SystemThroughput, self).__init__(efficiencies, wave=wave)


class VPHGratingEfficiency:
    """
    Calculate the nominal efficiency of a VPH grating.

    Based on IDL code provided Jason Fucik (CalTech), 7 Apr 2020.

    Args:
        line_density (:obj:`float`):
            Line density of the grating in lines per mm.
        n_bulk (:obj:`float`):
            Bulk index of refraction.
        n_mod (:obj:`float`):
            Modulation in the index of refraction.
        thickness (:obj:`float`):
            Grating thickness in micron.
        tilt (:obj:`float`):
            Grating tilt in degrees.
        peak_wave (:obj:`float`):
            Wavelength in angstrom at the peak grating efficiency.
            **This is never used.**
    """
    def __init__(self, line_density, n_bulk, n_mod, thickness, tilt, peak_wave):
        self.line_density = line_density
        self.n_bulk = n_bulk
        self.n_mod = n_mod
        self.thickness = thickness
        # Note that psi is positive when the angle of incidence > angle
        # of diffraction
        self.psi = numpy.radians(tilt) + numpy.pi/2.
        self.peak_wave = peak_wave

    def __call__(self, wave, aoi=None):
        """
        Calculate the efficiency at the provided wavelengths.

        Args:
            wave (:obj:`float`, `numpy.ndarray`_):
                Wavelengths in angstroms at which to calculate the
                efficiency.
            aoi (:obj:`float`):
                Angle of incidence onto the grating. If None, return
                the super-blaze function.

        Returns:
            :obj:`float`, `numpy.ndarray`_: The grating efficiency.
            The type returned matches the type provided for ``wave``.
        """
        # TODO: Perform type checking.  Allow list objects for wave.
        if aoi is None:
            # Return the super-blaze function
            bragg = numpy.radians(self.bragg_angle(wave))
            # Calculate the s- and p-polarization transmission efficiencies
            # Factors of 1e4 are to convert thickness from micron to angstrom
            s = numpy.square(numpy.sin(numpy.pi * self.n_mod * self.thickness * 1e4
                                       / wave / numpy.cos(bragg)))
            p = numpy.square(numpy.sin(numpy.pi * self.n_mod * self.thickness * 1e4
                                       * numpy.cos(2*bragg) / wave / numpy.cos(bragg)))
            return (s+p)/2  #, s, p

        # Return the Bragg envelopes
        _aoi = numpy.arcsin(numpy.sin(numpy.radians(aoi))/self.n_bulk)  # AOI at the DCG layer
        cos_aoi = numpy.cos(_aoi)
        cos_psi = numpy.cos(self.psi)
        tpnbulk = 2 * numpy.pi * self.n_bulk

        # Convert the units of line density from lines/mm to lines/angstrom
        _line_density = self.line_density / 1e7
        # Convert the units of thickness from micron to angstrom
        _thickness = self.thickness * 1e4

        # Calculate the s- and p-polarization transmission efficiencies
        k = 2 * numpy.pi * _line_density
        ct = cos_aoi - wave * k * cos_psi / tpnbulk
        eta = _thickness * (k * numpy.cos(_aoi - self.psi) - wave * k*k / 2 / tpnbulk) / ct / 2
        ks = numpy.pi * self.n_mod / wave
        vs = ks * _thickness / numpy.sqrt(ct * cos_aoi)
        vp = -ks * numpy.cos(2*(_aoi - self.psi)) * _thickness / numpy.sqrt(ct * cos_aoi)
        s = numpy.square(numpy.sin(numpy.sqrt(vs*vs + eta*eta))) / (1 + numpy.square(eta/vs))
        p = numpy.square(numpy.sin(numpy.sqrt(vp*vp + eta*eta))) / (1 + numpy.square(eta/vp))
        return (s+p)/2  #, s, p

    def bragg_angle(self, wave, sine_of=False):
        """
        Calculate the grating Bragg angle in degrees in the DCG
        layer.

        Args:
            wave (:obj:`float`):
                Wavelength at which to calculate the Bragg angle
                (angstroms in air).
            sine_of (:obj:`bool`, optional):
                Return the sine of the angle instead of the angle
                itself.

        Returns:
            :obj:`float`: The (sine of) the Bragg angle.
        """
        # Calculate the (sine of the) Bragg angle, where the angle of
        # incidence = the angle of diffraction, in the DCG layer. The
        # factor of 1e-7 converts the wavelength from angstroms to mm.
        sin_bragg = 1e-7*wave * self.line_density / 2 / self.n_bulk
        return sin_bragg if sine_of else numpy.degrees(numpy.arcsin(sin_bragg))

    def bragg_angle_of_incidence(self, wave):
        """
        Compute the super-blaze incidence angle for the grating.

        Args:
            wave (:obj:`float`):
                Wavelength at which to calculate the Bragg angle
                (angstroms in air).

        Returns:
            :obj:`float`: The super-blaze incidence angle.
        """
        # Return the angle of incidence using Snell's Law in air
        return numpy.degrees(numpy.arcsin(self.n_bulk * self.bragg_angle(wave, sine_of=True)))


class WFOSGratingEfficiency(VPHGratingEfficiency):
    """
    Object that computes the efficiency of the WFOS VPH gratings.

    Grating names and paramters provided by Jason Fucik (CalTech), 7
    Apr 2020.

    Args:
        grating (:obj:`str`):
            The grating name to use. Available gratings can be listed
            as follows::

                from enyo.etc.efficiency import WFOSGratingEfficiency
                print(list(WFOSGratingEfficiency.available_gratings.keys()))

    Raises:
        ValueError:
            Raised if the grating name is not recognized or entered
            as None.
    """

    available_gratings = dict(B1210=dict(line_density=1210., n_bulk=1.35, n_mod=0.05,
                                         thickness=4.0, tilt=0.0, peak_wave=3930.),
                              B2479=dict(line_density=2479., n_bulk=1.35, n_mod=0.13,
                                         thickness=1.742, tilt=0.0, peak_wave=4090.),
                              B2700=dict(line_density=2700., n_bulk=1.17, n_mod=0.17,
                                         thickness=3.50, tilt=0.0, peak_wave=4940.),
                              B3600=dict(line_density=3600., n_bulk=1.17, n_mod=0.15,
                                         thickness=3.03, tilt=0.0, peak_wave=3750.),
                              R680=dict(line_density=680., n_bulk=1.35, n_mod=0.07,
                                        thickness=5.35, tilt=0.0, peak_wave=7350.),
                              R1392=dict(line_density=1392., n_bulk=1.35, n_mod=0.14,
                                         thickness=2.85, tilt=0.0, peak_wave=7220.),
                              R1520=dict(line_density=1520., n_bulk=1.17, n_mod=0.23,
                                         thickness=4.67, tilt=0.0, peak_wave=8840.),
                              R2052=dict(line_density=2052., n_bulk=1.17, n_mod=0.20,
                                         thickness=4.01, tilt=0.0, peak_wave=6570.))
    """
    Provides the salient properties of the available WFOS gratings.
    """

    def __init__(self, grating):
        if grating is None:
            raise ValueError('Grating ID must be defined.')
        if grating not in self.available_gratings.keys():
            raise ValueError('Unknown grating ID ({0}).  Options are: {1}'.format(grating,
                                ', '.join(list(self.available_gratings.keys()))))
        super(WFOSGratingEfficiency, self).__init__(
                    self.available_gratings[grating]['line_density'],
                    self.available_gratings[grating]['n_bulk'],
                    self.available_gratings[grating]['n_mod'],
                    self.available_gratings[grating]['thickness'],
                    self.available_gratings[grating]['tilt'],
                    self.available_gratings[grating]['peak_wave'])


class TMTWFOSArmEfficiency:
    """
    Base object used to compute the total efficiency of each TMT-WFOS
    spectrograph arm. This class cannot be used directly and must be
    a base class; see :attr:`base_file` and :attr:`default_grating`.

    Based on IDL code provided Jason Fucik (CalTech), 7 Apr 2020.

    Args:
        cen_wave (:obj:`float`):
            Wavelength at the field center in angstroms.
        grating (:obj:`str`, optional):
            The grating name for this arm. If None, use
            :attr:`default_grating`.
        grating_angle (:obj:`float`, optional):
            The angle of the grating normal with respect to the
            incoming beam in degrees. If None, the grating Bragg
            angle is used; see
            :func:`VPHGratingEfficiency.bragg_angle_of_incidence`.

    Raises:
        NotImplementedError:
            Raised if the derived class (see, e.g.,
            :class:`TMTWFOSBlueEfficiency`) does not have
            :attr:`base_file` defined.

    """

    plate_scale = 2.18
    """TMT plate scale in units of mm per arcsecond."""

    focal_length = 4500.
    """WFOS collimator focal length in mm"""

    base_file = None
    """
    File with the grating-independent efficiencies. Must be defined
    by the derived class.
    """

    default_grating = None
    """
    Default grating to use for this arm. Must be defined by the
    derived class.
    """

    def __init__(self, cen_wave, grating=None, grating_angle=None):
        self.cen_wave = cen_wave
        self.grating_eff = WFOSGratingEfficiency(grating=self.default_grating
                                                    if grating is None else grating)
        self.grating_angle = self.grating_eff.bragg_angle_of_incidence(self.cen_wave) \
                                if grating_angle is None else grating_angle
        self.base_efficiency = self._base_efficiency()

    def _base_efficiency(self):
        """
        Internal method to construct the baseline efficiency (i.e.,
        all the efficiencies of all the spectrograph components for
        this arm, except for this grating).
        """
        if self.base_file is None:
            raise NotImplementedError('Must define the base_file for {0}'.format(
                                        self.__class__.__name__))
        db = numpy.genfromtxt(self.base_file)
        # Wavelengths in these files are expected to be in nm.
        # Wavelengths are expected to be in the first column. All other
        # columns are expected to be efficiencies of all other
        # instrument components, the product of which provides the
        # total efficiency of everything but the grating.
        return Efficiency(numpy.prod(db[:,1:], axis=1), wave=db[:,0]*10)

    def __call__(self, wave, x=0.0, to_efficiency=False):
        """
        Compute the total efficiency for this arm, from the ADC
        through to (and including) the detector.

        .. note::

            - For the moment this does not include the y position in
              the slit mask.

            - Note that x must be a scalar for now.

            - Method may fail if ``wave`` is a single value and
            ``to_efficiency`` is True.

        Args:
            wave (:obj:`float`, `numpy.ndarray`_):
                One or more wavelengths (angstroms in air) at which
                to sample the spectrograph efficiency.
            x (:obj:`float`, optional):
                Field position perpendicular to the dispersion direction
                in arcseconds relative to the field center.
            to_efficiency (:obj:`bool`, optional):
                Instead of returning the efficiency values only,
                return an :class:`Efficiency` object.

        Returns:
            :obj:`float`, `numpy.ndarray`_, :class:`Efficiency`: The
            efficiency of the spectrograph arm at one or more
            wavelengths. The returned type matches the input type of
            ``wave``.
        """
        # TODO: Check behavior of arctan vs. arctan2.
        angle = self.grating_angle \
                    + numpy.degrees(numpy.arctan(x * self.plate_scale / self.focal_length))
        eta = self.base_efficiency(wave) * self.grating_eff(wave, aoi=angle)
        return Efficiency(eta, wave=wave) if to_efficiency else eta


class TMTWFOSBlueEfficiency(TMTWFOSArmEfficiency):
    """
    Object used to compute the efficiency of the blue arm of the
    TMT-WFOS spectrograph. See the base class for the description of
    the instantiation arguments and methods.
    """

    base_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'efficiency',
                             'wfos_blue_efficiency.db')
    """
    The file containing tabulated efficiency data for the elements
    of the spectrograph arm, excluding the grating.
    """

    default_grating = 'B1210'
    """
    The default grating used when one is not specified at
    instantiation.
    """


class TMTWFOSRedEfficiency(TMTWFOSArmEfficiency):
    """
    Object used to compute the efficiency of the red arm of the
    TMT-WFOS spectrograph. See the base class for the description of
    the instantiation arguments and methods.
    """

    base_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'efficiency',
                             'wfos_red_efficiency.db')
    """
    The file containing tabulated efficiency data for the elements
    of the spectrograph arm, excluding the grating.
    """

    default_grating = 'R680'
    """
    The default grating used when one is not specified at
    instantiation.
    """
        

