import os
import warnings

import numpy

from matplotlib import pyplot

from astropy.io import fits

from enyo.etc import source, telescopes, aperture, spectrum, efficiency

def main():
    manga_sky_file = os.path.join(os.environ['ENYO_DIR'],
                                  'data/sky/manga/mgCFrame-00183643-LOG.fits.gz')
    deimos_sky_file = os.path.join(os.environ['ENYO_DIR'],
                                   'data/sky/mkea_sky_newmoon_DEIMOS_1200_2011oct.fits.gz')
    scaled_sky_file = os.path.join(os.environ['ENYO_DIR'],
                                   'data/sky/manga/apo2maunakeasky.fits')

    hdu = fits.open(scaled_sky_file)
    pyplot.plot(hdu['WAVE'].data, hdu['FLUX'].data)
    hdu.close()
    del hdu

    hdu = fits.open(manga_sky_file)
    manga_wave = hdu['WAVE'].data
    manga_sky_flux = numpy.ma.MaskedArray(hdu['SKY'].data, mask=hdu['MASK'].data > 0)
    manga_sky_flux = numpy.ma.median(manga_sky_flux, axis=0)/numpy.pi
    hdu.close()
    del hdu

#    pyplot.imshow(numpy.ma.log10(sky_flux/numpy.pi), origin='lower', interpolation='nearest', aspect='auto')
#    pyplot.colorbar()
#    pyplot.show()

    hdu = fits.open(deimos_sky_file)
    deimos_wave = hdu[0].data
#    deimos_sky_flux = numpy.ma.MaskedArray(hdu[1].data, mask=hdu[1].data > 0)
    deimos_sky_flux = hdu[1].data*1e17
    hdu.close()
    del hdu

    pyplot.plot(manga_wave, manga_sky_flux)
    pyplot.plot(deimos_wave, deimos_sky_flux)
    pyplot.show()


if __name__ == '__main__':
    main()