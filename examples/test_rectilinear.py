import os
import numpy
from scipy import stats

from matplotlib import pyplot

from astropy.io import fits

from enyo.etc import observe, source, efficiency, spectrum, aperture, spectrographs

def simple():
    wave = numpy.arange(100)
    flux = stats.norm.pdf(wave, loc=50, scale=5)
    spec = Spectrum(wave, flux)
#    spec.show()

    slit = aperture.SlitAperture(0, 0, 3, 10)
    x = numpy.arange(15)-15//2
    y = numpy.arange(15)-15//2
    slit_img = slit.response(x,y)

#    pyplot.imshow(slit_img, origin='lower', interpolation='nearest')
#    pyplot.show()

    test_img = observe.rectilinear_twod_spectrum(spec, slit_img, 1)
    pyplot.imshow(test_img, origin='lower', interpolation='nearest')
    pyplot.show()

def direct():

    seeing = 0.6            # arcsec
    wave_ref = 5500         # Reference wavelength in angstrom
    slitwidth = 0.75        # arcsec
    slitlength = 5.         # arcsec
    slitrotation = 0.       # degrees

    sky_spec = spectrum.MaunakeaSkySpectrum()
#    pyplot.plot(sky_spec.wave, sky_spec.flux)
#    pyplot.show()

    psf = source.OnSkyGaussian(seeing)
    
    sky_reference_flux = sky_spec.interp([wave_ref])[0]
    sky = source.OnSkySource(psf, source.OnSkyConstant(sky_reference_flux))

    slit = aperture.SlitAperture(0., 0., slitwidth, slitlength, rotation=slitrotation)

    lowres_wfos = spectrographs.TMTWFOS(setting='lowres')
    off_slit_img = lowres_wfos.monochromatic_image(sky, slit, arm='blue')

#    pyplot.imshow(off_slit_img, origin='lower', interpolation='nearest')
#    pyplot.show()

    nspec,nspat = off_slit_img.shape
#    pyplot.plot(off_slit_img[:,nspat//2])
#    pyplot.show()

    test_twod = observe.rectilinear_twod_spectrum(sky_spec, off_slit_img,
                                                  lowres_wfos.arms['blue'].dispscale)
    print('finished')

    pyplot.imshow(test_twod, origin='lower', interpolation='nearest', aspect='auto')
    pyplot.show()

def main():
    op = spectrographs.TMTWFOSBlueOpticalModel()
    op.field2camera(0, 0, 0.4000)

def twod_spec():

    mag = 23.               # g-band AB magnitude
    seeing = 0.6            # arcsec
    wave_ref = 5500         # Reference wavelength in angstrom
    slitwidth = 0.75        # arcsec
    slitlength = 5.         # arcsec
    slitrotation = 0.       # degrees

#    wave = numpy.linspace(3100., 10000., num=6901, dtype=float)
#    star_spec = spectrum.ABReferenceSpectrum(wave)
    star_spec = spectrum.BlueGalaxySpectrum()

    star_spec.show()

    g = efficiency.FilterResponse()
    star_spec.rescale_magnitude(g, mag)

    sky_spec = spectrum.MaunakeaSkySpectrum()

    psf = source.OnSkyGaussian(seeing)
    star_reference_flux = star_spec.interp([wave_ref])[0]
    star = source.OnSkySource(psf, star_reference_flux)    

    sky_reference_flux = sky_spec.interp([wave_ref])[0]
    sky = source.OnSkySource(psf, source.OnSkyConstant(sky_reference_flux))

    slit = aperture.SlitAperture(0., 0., slitwidth, slitlength, rotation=slitrotation)

    lowres_wfos = spectrographs.TMTWFOS(setting='lowres')

    wave_lim = star_spec.wave[[0,-1]]

    arm = 'red'

    print('Building off-slit rectilinear spectrum')
    off_slit_spectrum = lowres_wfos.twod_spectrum(sky_spec, slit, wave_lim=wave_lim, arm=arm,
                                                  rectilinear=True)

#    print(numpy.median(off_slit_spectrum))
#    pyplot.imshow(off_slit_spectrum, origin='lower', interpolation='nearest', aspect='auto',
#                  vmax=10)
#    pyplot.show()

    print('Building on-slit rectilinear spectrum')
    on_slit_spectrum = lowres_wfos.twod_spectrum(sky_spec, slit, source_distribution=star,
                                                 source_spectrum=star_spec, wave_lim=wave_lim,
                                                 arm=arm, rectilinear=True)
#    pyplot.imshow(on_slit_spectrum, origin='lower', interpolation='nearest', aspect='auto',
#                  vmax=10)
#    pyplot.show()

    print('Writing rectilinear spectra to disk.')
    fits.HDUList([fits.PrimaryHDU(data=off_slit_spectrum)]).writeto('offslit_test.fits',
                                                                    overwrite=True)
    fits.HDUList([fits.PrimaryHDU(data=on_slit_spectrum)]).writeto('onslit_test.fits',
                                                                   overwrite=True)


    # Field coordinates in arcsec
    field_coo=numpy.array([-4.,1.2])*60.

    print('Building off-slit projected spectrum.')
    off_slit_projected_spectrum, spec0, spat0 \
            = lowres_wfos.twod_spectrum(sky_spec, slit, wave_lim=wave_lim, arm=arm,
                                        field_coo=field_coo)

#    from IPython import embed
#    embed()

    print('Building on-slit projected spectrum.')
    on_slit_projected_spectrum, spec0, spat0 \
            = lowres_wfos.twod_spectrum(sky_spec, slit, source_distribution=star,
                                        source_spectrum=star_spec, wave_lim=wave_lim, arm=arm,
                                        field_coo=field_coo)

#    embed()
    print(on_slit_projected_spectrum.shape)

    fits.HDUList([fits.PrimaryHDU(data=off_slit_projected_spectrum)
                    ]).writeto('offslit_projected_test.fits', overwrite=True)
    fits.HDUList([fits.PrimaryHDU(data=on_slit_projected_spectrum)
                  ]).writeto('onslit_projected_test.fits', overwrite=True)

#    lowres_wfos.
#
#    fits.HDUList([fits.PrimaryHDU(data=on_slit_spectrum)]).writeto('onslit_test_projected.fits',
#                                                                   overwrite=True)


def gal_set_image():

    seeing = 0.6            # arcsec
    wave_ref = 5500         # Reference wavelength in angstrom
    slitwidth = 0.75        # arcsec
    slitlength = 5.         # arcsec
    slitrotation = 0.       # degrees

    slit_pos_file = 'slits.db'
    if not os.path.isfile(slit_pos_file):
        slit_x = numpy.arange(-4.2*60+slitlength, 4.2*60, slitlength+slitlength*0.2)
        nslits = slit_x.size
        slit_y = numpy.random.uniform(size=nslits)*3-1.5
        mag = 23-numpy.random.uniform(size=nslits)*2
        z = 0.1 + numpy.random.uniform(size=nslits)*0.4
        numpy.savetxt(slit_pos_file, numpy.array([slit_x, slit_y, mag, z]).T,
                      header='{0:>7} {1:>7} {2:>5} {3:>7}'.format('SLITX', 'SLITY', 'GMAG', 'Z'),
                      fmt=['%9.2f', '%7.2f', '%5.2f', '%7.5f'])
    else:
        slit_x, slit_y, mag, z = numpy.genfromtxt(slit_pos_file).T
        nslits = slit_x.size

    nspec = 18000
    nspat = 10800
    image = {'blue': numpy.zeros((nspec,nspat), dtype=numpy.float32),
             'red': numpy.zeros((nspec,nspat), dtype=numpy.float32) }

    psf = source.OnSkyGaussian(seeing)
    g = efficiency.FilterResponse()

    sky_spec = spectrum.MaunakeaSkySpectrum()
    sky_reference_flux = sky_spec.interp([wave_ref])[0]
    sky = source.OnSkyConstant(sky_reference_flux)

    slit = aperture.SlitAperture(0., 0., slitwidth, slitlength, rotation=slitrotation)
    lowres_wfos = spectrographs.TMTWFOS(setting='lowres')
    wave_lim = numpy.array([3100, 10000])

    gal = source.OnSkySource(psf, source.OnSkySersic(1.0, 0.1, 1, unity_integral=True),
                             sampling=lowres_wfos.arms['blue'].pixelscale)
    
#    pyplot.imshow(gal.data, origin='lower', interpolation='nearest')
#    pyplot.show()

    for i in range(nslits):
        gal_spec = spectrum.BlueGalaxySpectrum()
        gal_spec.redshift(z[i])
        gal_spec.rescale_magnitude(g, mag[i])

        field_coo = numpy.array([slit_x[i], slit_y[i]])

        for arm in ['blue', 'red']:
            print('Building on-slit projected spectrum.')
            slit_spectrum, spec0, spat0 \
                    = lowres_wfos.twod_spectrum(sky_spec, slit, source_distribution=gal,
                                                source_spectrum=gal_spec, wave_lim=wave_lim,
                                                arm=arm, field_coo=field_coo)
            sspec = int(spec0)+nspec//2
            sspat = int(spat0)+nspat//2

            image[arm][sspec:sspec+slit_spectrum.shape[0], sspat:sspat+slit_spectrum.shape[1]] \
                    += slit_spectrum

    fits.HDUList([fits.PrimaryHDU(data=image['blue'])]).writeto('gal_set_blue.fits', overwrite=True)
    fits.HDUList([fits.PrimaryHDU(data=image['red'])]).writeto('gal_set_red.fits', overwrite=True)


if __name__ == '__main__':
    gal_set_image()

