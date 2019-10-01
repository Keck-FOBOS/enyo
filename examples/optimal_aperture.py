import os
import warnings
import time

import numpy
from scipy import interpolate

from astropy.io import fits

from matplotlib import pyplot, ticker, rc, colors, cm, colorbar, image, patches

from enyo.etc import source, telescopes, aperture, spectrum, efficiency, util

def test(mag, seeing, on_sky_fiber_diameter, sersic=None, exptime=1.):
    telescope = telescopes.KeckTelescope()
    detector_noise_ratio = 0. #1/3.
    re, sersicn = (None,None) if sersic is None else sersic
    size = 5.0 if re is None else max(5.0, 8*re)
    sampling = 0.05 if re is None else min(0.05, re/20)    # arcsec/pixel
    return onsky_aperture_snr(telescope, mag, seeing, on_sky_fiber_diameter,
                              detector_noise_ratio=detector_noise_ratio, sampling=sampling,
                              size=size, re=re, sersicn=sersicn, exptime=exptime, quiet=True)

def build_data(mag, ofile):

#    dmag_sky = None     # Dark time
#
#    dmag_sky = -3.0     # Bright time...

    seeing = [0.4, 0.6, 0.8, 1.0]
    sersic = [None, (0.1,1), (0.1,4), (0.2,1), (0.2,4), (0.4,1), (0.4,4), (0.8,1), (0.8,4)]
#    seeing = [0.4, 0.6]
#    sersic = [None]

    telescope = telescopes.KeckTelescope()
    dmag_sky = None
    fobos_fratio = 3.
    detector_noise_ratio = 0. #1/3.

    fiber_diameter = numpy.linspace(0.05, 0.25, 41)
    fobos_platescale = telescope.platescale*fobos_fratio/telescope.fratio

    outshape = (len(seeing), len(sersic), len(fiber_diameter))

    atmseeing = numpy.zeros(outshape, dtype=float)
    profile = numpy.zeros(outshape, dtype=int)
    fibdiam = numpy.zeros(outshape, dtype=float)
    total_snr = numpy.zeros(outshape, dtype=float)
    skysub_snr = numpy.zeros(outshape, dtype=float)

    for i in range(len(seeing)):
        atmseeing[i,:,:] = seeing[i]
        for j in range(len(sersic)):
            print('Seeing {0}/{1}'.format(i+1, len(seeing)))
            print('Profile {0}/{1}'.format(j+1, len(sersic)))

            profile[i,j,:] = j

            if sersic[j] is None:
                re = None
                sersicn = None
            else:
                re, sersicn = sersic[j]

            size = 5.0 if re is None else max(5.0, 8*re)
            sampling = 0.05 if re is None else min(0.05, re/20)    # arcsec/pixel

            for k in range(len(fiber_diameter)):
                fibdiam[i,j,k] = fiber_diameter[k]
                print('Calculating {0}/{1}'.format(k+1, fiber_diameter.size), end='\r')
                total_snr[i,j,k], skysub_snr[i,j,k] = \
                    aperture_snr(telescope, mag, seeing[i], fiber_diameter[k], dmag_sky=dmag_sky,
                                 fobos_fratio=fobos_fratio,
                                 detector_noise_ratio=detector_noise_ratio, sampling=sampling,
                                 size=size, re=re, sersicn=sersicn, quiet=True)
            print('Calculating {0}/{0}'.format(fiber_diameter.size))

    fits.HDUList([ fits.PrimaryHDU(),
                   fits.ImageHDU(data=atmseeing, name='SEEING'),
                   fits.ImageHDU(data=profile, name='PROF'),
                   fits.ImageHDU(data=fibdiam, name='DIAM'),
                   fits.ImageHDU(data=total_snr, name='TOTAL'),
                   fits.ImageHDU(data=skysub_snr, name='SKYSUB')
                 ]).writeto(ofile, overwrite=True)
                   
    return

    i = numpy.argmax(total_snr)
    print('Max total: {0:.3f} {1:.3f}'.format(fiber_diameter[i], total_snr[i]))
    i = numpy.argmax(skysub_snr)
    print('Max skysub: {0:.3f} {1:.3f}'.format(fiber_diameter[i], skysub_snr[i]))

    print('Calculating {0}/{0}'.format(fiber_diameter.size))
    pyplot.plot(fiber_diameter*1000, total_snr)
    pyplot.plot(fiber_diameter*1000, skysub_snr)
    pyplot.xlabel('Fiber Diameter (micron)')
    pyplot.ylabel(r'S/N [(photons/s) / $\sqrt{{\rm photons}^2/{\rm s}^2}$]')
    pyplot.show()

def aperture_snr(telescope, mag, seeing, fiber_diameter, dmag_sky=None, fobos_fratio=3.0,
                 detector_noise_ratio=1., sampling=0.1, size=5.0, re=None, sersicn=None,
                 exptime=1., quiet=False):

    # FOBOS focal-plane platescale (arcsec/mm)
    fobos_platescale = telescope.platescale*fobos_fratio/telescope.fratio
    on_sky_fiber_diameter = fiber_diameter/fobos_platescale   # arcsec
    return onsky_aperture_snr(telescope, mag, seeing, on_sky_fiber_diameter, dmag_sky=dmag_sky,
                              detector_noise_ratio=detector_noise_ratio, sampling=sampling,
                              size=size, re=re, sersicn=sersicn, exptime=exptime, quiet=quiet)


def onsky_aperture_snr(telescope, mag, seeing, on_sky_fiber_diameter, dmag_sky=None,
                       detector_noise_ratio=1., sampling=0.1, size=5.0, re=None, sersicn=None,
                       exptime=1., quiet=False):

    # Use a reference spectrum that's constant; units are 1e-17 erg/s/cm^2/angstrom
    wave = numpy.linspace(3000., 10000., num=7001)
    spec = spectrum.ABReferenceSpectrum(wave)
    g = efficiency.FilterResponse()
    spec.rescale_magnitude(g, mag)

    # Sky Spectrum; units are 1e-17 erg/s/cm^2/angstrom/arcsec^2
    sky = spectrum.MaunakeaSkySpectrum()
    sky_mag = sky.magnitude(g)
    if dmag_sky is not None:
        sky_mag += dmag_sky
        sky.rescale_magnitude(g, sky_mag)

#    # For the image representation of the fiber and source, use a fixed
#    # sampling
#    size = (int(numpy.ceil(on_sky_fiber_diameter*1.2/sampling))+1)*sampling

    # Build the source
    intrinsic = 1.0 if re is None or sersicn is None \
                    else source.OnSkySersic(1.0, re, sersicn, sampling=sampling, size=size,
                                            unity_integral=True)
    src = source.OnSkySource(seeing, intrinsic, sampling=sampling, size=size)
    if numpy.absolute(numpy.log10(numpy.sum(src.data*numpy.square(src.sampling)))) > 0.1:
        raise ValueError('Bad source representation; image integral is {0:.3f}.'.format(
                            numpy.sum(src.data*numpy.square(src.sampling))))
    # Define fiber aperture
    fiber = aperture.FiberAperture(0., 0., on_sky_fiber_diameter, resolution=100)
    # Generate its response function
    fiber_response = fiber.response(src.x, src.y)
    in_fiber = numpy.sum(src.data*fiber_response)*numpy.square(src.sampling)

    # Scale the fiber response profile to produce a uniformly scrambled
    # fiber output beam
    scale = numpy.sum(src.data*fiber_response)/numpy.sum(fiber_response)
    source_fiber_img = scale*fiber_response

    # Convert them to photons/angstrom (per arcsec^2 for sky)
    spec.rescale(telescope.area*exptime)
    sky.rescale(telescope.area*exptime)
    spec.photon_flux()
    sky.photon_flux()

    # Get the total number of source and sky photons at
    # the detector integrated over the g-band filter
    flux = numpy.sum(spec.flux * g(spec.wave) * spec.wavelength_step())
    sky_flux = numpy.sum(sky.flux * g(sky.wave) * sky.wavelength_step())

    # Scale the source image by the expected flux (source_fiber_img was
    # constructed assuming unity flux)
    fiber_obj_img = source_fiber_img * flux
    # Scale the fiber response function by the sky surface brightness
    # (the integral of the sky image is then just the fiber area times
    # the uniform sky surface brightness)
    fiber_sky_img = fiber_response * sky_flux

    # Integrate spatially over the pixel size to get the number of
    # photons/s/cm^2 entering the fiber
    fiber_obj_flux = numpy.sum(fiber_obj_img * numpy.square(src.sampling))
    fiber_sky_flux = numpy.sum(fiber_sky_img * numpy.square(src.sampling))

    total_flux = fiber_obj_flux + fiber_sky_flux
    sky_var = fiber_sky_flux + detector_noise_ratio*total_flux
    total_var = fiber_obj_flux + sky_var
    total_snr = fiber_obj_flux/numpy.sqrt(total_var)
    skysub_snr = fiber_obj_flux/numpy.sqrt(total_var + sky_var)

    if not quiet:
        print('Star Magnitude (AB mag): {0:.2f}'.format(spec.magnitude(g)))
        print('Sky Surface Brightness (AB mag/arcsec^2): {0:.2f}'.format(sky_mag))
        print('On-sky fiber diameter (arcsec): {0:.2f}'.format(on_sky_fiber_diameter))
        print('Pixel sampling of maps: {0:.3f} (arcsec/pixel)'.format(sampling))
        print('Square map size : {0:.3f} (arcsec)'.format(size))
        print('Fraction of source flux in mapped image: {0:.3f}'.format(numpy.sum(src.data)
                                                                * numpy.square(sampling)))
        print('Fiber area (mapped): {0:.2f} (arcsec^2)'.format(numpy.sum(fiber_response)
                                                                * numpy.square(src.sampling)))
        print('Fiber area (nominal): {0:.2f} (arcsec^2)'.format(fiber.area))
        print('Aperture loss: {0:.3f}'.format(in_fiber))
        print('Object flux: {0:.3e} (g-band photons/s)'.format(fiber_obj_flux))
        print('Sky flux: {0:.3e} (g-band photons/s)'.format(fiber_sky_flux))
        print('Total S/N: {0:.3f}'.format(total_snr))
        print('Sky-subtracted S/N: {0:.3f}'.format(skysub_snr))

#    return fiber_obj_flux, fiber_sky_flux, total_snr, skysub_snr
    return total_snr, skysub_snr


def source_plot(ax, hdu, j):

    telescope = telescopes.KeckTelescope()
    fobos_fratio = 3.
    fobos_platescale = telescope.platescale*fobos_fratio/telescope.fratio

    fiber_diameter_limits = numpy.array([48., 252.])
    ax.set_xlim(fiber_diameter_limits)
    ax.set_ylim([0.0, 6.])
    ax.minorticks_on()
    ax.tick_params(which='major', length=8, direction='in')
    ax.tick_params(which='minor', length=4, direction='in')
    ax.grid(True, which='major', color='0.9', linestyle=':')
    for i in range(4):
        ax.plot(hdu['DIAM'].data[i,j,:]*1000, hdu['TOTAL'].data[i,j,:],
                color='C{0}'.format(i), linestyle='-', zorder=2)
        ax.plot(hdu['DIAM'].data[i,j,:]*1000, hdu['SKYSUB'].data[i,j,:],
                color='C{0}'.format(i), linestyle='--', zorder=2)
        m = numpy.array([numpy.argmax(hdu['TOTAL'].data[i,j,:]),
                         numpy.argmax(hdu['SKYSUB'].data[i,j,:])])
        maxsnr = numpy.array([hdu['TOTAL'].data[i,j,m[0]], 
                              hdu['SKYSUB'].data[i,j,m[1]]])
        ax.scatter(hdu['DIAM'].data[i,j,m]*1000, maxsnr,
                   marker='.', color='C{0}'.format(i), lw=0, s=100, zorder=3)
    
    ax2 = ax.twiny()
    ax2.minorticks_on()
    ax2.tick_params(which='major', length=8, direction='in')
    ax2.tick_params(which='minor', length=4, direction='in')
    ax2.set_xlim(fiber_diameter_limits/fobos_platescale/1000)

    return ax, ax2


def make_plots(ofile):
    
    hdu = fits.open(ofile)
   
    rc('font', size=10)
    w,h = pyplot.figaspect(1)

    fig = pyplot.figure(figsize=(1.5*w,1.5*h))

    # Labels
    ax = fig.add_axes([0.08, 0.46, 0.25, 0.2], frameon=False)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.text(0.2, 0.90, 'Seeing =', ha='left', va='center', transform=ax.transAxes)
    ax.axhline(0.8, xmin=0.2, xmax=0.4, color='C0')
    ax.axhline(0.77, xmin=0.2, xmax=0.4, color='C0', linestyle='--')
    ax.text(0.42, 0.785, '0.4 arcsec', ha='left', va='center', transform=ax.transAxes)
    ax.axhline(0.67, xmin=0.2, xmax=0.4, color='C1')
    ax.axhline(0.64, xmin=0.2, xmax=0.4, color='C1', linestyle='--')
    ax.text(0.42, 0.655, '0.6 arcsec', ha='left', va='center', transform=ax.transAxes)
    ax.axhline(0.54, xmin=0.2, xmax=0.4, color='C2')
    ax.axhline(0.51, xmin=0.2, xmax=0.4, color='C2', linestyle='--')
    ax.text(0.42, 0.525, '0.8 arcsec', ha='left', va='center', transform=ax.transAxes)
    ax.axhline(0.41, xmin=0.2, xmax=0.4, color='C3')
    ax.axhline(0.38, xmin=0.2, xmax=0.4, color='C3', linestyle='--')
    ax.text(0.42, 0.395, '1.0 arcsec', ha='left', va='center', transform=ax.transAxes)
    ax.axhline(0.20, xmin=0.1, xmax=0.3, color='k')
    ax.text(0.32, 0.20, 'Best S/N', ha='left', va='center', transform=ax.transAxes)
    ax.axhline(0.10, xmin=0.1, xmax=0.3, color='k', linestyle='--')
    ax.text(0.32, 0.10, 'after sky-sub', ha='left', va='center', transform=ax.transAxes)

    ax, ax2 = source_plot(fig.add_axes([0.1, 0.74, 0.25, 0.2]), hdu, 0)
    ax.axvline(x=150., color='k', linestyle='--', lw=0.5)
    ax.text(0.5, -0.2, r'Fiber Diameter [$\mu$m]', ha='center', va='center',
            transform=ax.transAxes)
    ax.text(-0.2, 0.5, r'S/N [$\gamma$/s / $\sqrt{\gamma^2/{\rm s}^2}$]',
            ha='center', va='center', transform=ax.transAxes, rotation='vertical')
    ax.text(0.99, 0.85, r'Point Source', ha='right', va='center',
            transform=ax.transAxes)

    ax, ax2 = source_plot(fig.add_axes([0.37, 0.74, 0.25, 0.2]), hdu, 1)
    ax.axvline(x=150., color='k', linestyle='--', lw=0.5)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax2.text(0.5, 1.2, r'Fiber Diameter [arcsec]', ha='center', va='center',
             transform=ax2.transAxes)
    ax.text(0.99, 0.85, r'Sersic, $n=1$', ha='right', va='center',
            transform=ax.transAxes)

    ax, ax2 = source_plot(fig.add_axes([0.64, 0.74, 0.25, 0.2]), hdu, 2)
    ax.axvline(x=150., color='k', linestyle='--', lw=0.5)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.text(1.07, 0.5, r'$R_{eff}=0.1$ arcsec',
            ha='center', va='center', transform=ax.transAxes, rotation=-90)
    ax.text(0.99, 0.85, r'Sersic, $n=4$', ha='right', va='center',
            transform=ax.transAxes)

    ax, ax2 = source_plot(fig.add_axes([0.37, 0.52, 0.25, 0.2]), hdu, 3)
    ax.axvline(x=150., color='k', linestyle='--', lw=0.5)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())

    ax, ax2 = source_plot(fig.add_axes([0.64, 0.52, 0.25, 0.2]), hdu, 4)
    ax.axvline(x=150., color='k', linestyle='--', lw=0.5)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.text(1.07, 0.5, r'$R_{eff}=0.2$ arcsec',
            ha='center', va='center', transform=ax.transAxes, rotation=-90)

    ax, ax2 = source_plot(fig.add_axes([0.37, 0.30, 0.25, 0.2]), hdu, 5)
    ax.axvline(x=150., color='k', linestyle='--', lw=0.5)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.text(-0.2, 0.5, r'S/N [$\gamma$/s / $\sqrt{\gamma^2/{\rm s}^2}$]',
            ha='center', va='center', transform=ax.transAxes, rotation='vertical')

    ax, ax2 = source_plot(fig.add_axes([0.64, 0.30, 0.25, 0.2]), hdu, 6)
    ax.axvline(x=150., color='k', linestyle='--', lw=0.5)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.text(1.07, 0.5, r'$R_{eff}=0.4$ arcsec',
            ha='center', va='center', transform=ax.transAxes, rotation=-90)

    ax, ax2 = source_plot(fig.add_axes([0.37, 0.08, 0.25, 0.2]), hdu, 7)
    ax.axvline(x=150., color='k', linestyle='--', lw=0.5)
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.text(0.5, -0.2, r'Fiber Diameter [$\mu$m]', ha='center', va='center',
            transform=ax.transAxes)

    ax, ax2 = source_plot(fig.add_axes([0.64, 0.08, 0.25, 0.2]), hdu, 8)
    ax.axvline(x=150., color='k', linestyle='--', lw=0.5)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.text(0.5, -0.2, r'Fiber Diameter [$\mu$m]', ha='center', va='center',
            transform=ax.transAxes)
    ax.text(1.07, 0.5, r'$R_{eff}=0.8$ arcsec',
            ha='center', va='center', transform=ax.transAxes, rotation=-90)

    fig.canvas.print_figure('optimal_aperture.pdf', bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)
#    pyplot.show()

#    for j in range(2,9,2):
#        pyplot.show()
#        fig.clear()
#        pyplot.close(fig)


def test_mag_diff():
    seeing = 0.8
    mag = numpy.linspace(19.,29.,20)
    skysubsnr_star = numpy.zeros(mag.size, dtype=float)
    skysubsnr_disk = numpy.zeros(mag.size, dtype=float)
    skysubsnr_elli = numpy.zeros(mag.size, dtype=float)

#    for i in range(len(mag)):
    for i in range(len(t)):
        print('Mag: {0}/{1}'.format(i+1,mag.size), end='\r')
        skysubsnr_star[i] = test(mag[i], seeing, exptime=t[i])[1]
        skysubsnr_disk[i] = test(mag[i], seeing, sersic=(0.4,1), exptime=t[i])[1]
        skysubsnr_elli[i] = test(mag[i], seeing, sersic=(0.4,4), exptime=t[i])[1]

    pyplot.scatter(mag, skysubsnr_star, marker='.', s=60, lw=0, color='C0')
    pyplot.scatter(mag, skysubsnr_disk, marker='.', s=60, lw=0, color='C1')
    pyplot.scatter(mag, skysubsnr_elli, marker='.', s=60, lw=0, color='C3')
    pyplot.yscale('log')
    pyplot.show()


def test_time_diff():
    seeing = 0.8
    mag = 24.
    t = numpy.logspace(0,2,10)
    skysubsnr_star = numpy.zeros(t.size, dtype=float)
    skysubsnr_disk = numpy.zeros(t.size, dtype=float)
    skysubsnr_elli = numpy.zeros(t.size, dtype=float)

    for i in range(len(t)):
        print('Exp: {0}/{1}'.format(i+1,t.size), end='\r')
        skysubsnr_star[i] = test(mag, seeing, exptime=t[i])[1]
        skysubsnr_disk[i] = test(mag, seeing, sersic=(0.4,1), exptime=t[i])[1]
        skysubsnr_elli[i] = test(mag, seeing, sersic=(0.4,4), exptime=t[i])[1]

    pyplot.scatter(t, skysubsnr_star, marker='.', s=60, lw=0, color='C0')
    pyplot.scatter(t, skysubsnr_disk, marker='.', s=60, lw=0, color='C1')
    pyplot.scatter(t, skysubsnr_elli, marker='.', s=60, lw=0, color='C3')
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.plot([1,100], numpy.sqrt([1,100])*2, color='0.5')
    pyplot.show()


def makemap():
    mag = 24.
    fiberap = numpy.linspace(0.3,1.6,27)
    seeing = numpy.linspace(0.3,1.35,22)

    print(fiberap)
    print(seeing)

    F = numpy.zeros((fiberap.size, seeing.size), dtype=float)
    S = numpy.zeros((fiberap.size, seeing.size), dtype=float)
    snr = numpy.zeros((3,fiberap.size, seeing.size), dtype=float)
    sersic = [None, (0.4,1), (0.4,4)]
    for k in range(len(sersic)):
        for i in range(fiberap.size):
            for j in range(seeing.size):
                print('Cell: {0}/{1}'.format(i*seeing.size + j+1, snr.size), end='\r')
                snr[k,i,j] = test(mag, seeing[j], fiberap[i], sersic=sersic[k])[1]
    print('Cell: {0}/{0}'.format(snr.size))

    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=snr, name='SNR')]).writeto('test.fits',
                                                                                   overwrite=True)

if __name__ == '__main__':

    fiberap = numpy.linspace(0.3,1.6,27)
    seeing, count = numpy.genfromtxt('keck_seeing.db').T
    frac = count/numpy.sum(count)*0.05

    X,Y = numpy.meshgrid(seeing, fiberap)

#    hdu = fits.open('test.fits')
#    maxcoo = numpy.unravel_index(numpy.argmax(hdu['SNR'].data[0]), hdu['SNR'].data[0].shape)
#    print(maxcoo)
#    print(X[maxcoo], Y[maxcoo], hdu['SNR'].data[0][maxcoo])
#
#    pyplot.imshow(hdu['SNR'].data[0], origin='lower', interpolation='nearest')
#    pyplot.show()

#    extent=[seeing[0]-0.025, seeing[-1]+0.025, fiberap[0]-0.025, fiberap[-1]+0.025]
#    r = (extent[1]-extent[0])/(extent[3]-extent[2])
    extent=[fiberap[0]-0.025, fiberap[-1]+0.025, seeing[0]-0.025, seeing[-1]+0.025]
    hdu = fits.open('test.fits')

    print(numpy.amin(hdu['SNR'].data), numpy.amax(hdu['SNR'].data))
    img = numpy.square(hdu['SNR'].data)*frac[None,None,:]
    print(numpy.amin(img), numpy.amax(img))
    img = numpy.square(hdu['SNR'].data)*frac[None,None,:]/fiberap[None,:,None]
    print(numpy.amin(img), numpy.amax(img))

    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))

    ax = fig.add_axes([0.1, 0.73, 0.28, 0.21])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:])
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.imshow(hdu['SNR'].data[0].T,
              origin='lower', interpolation='nearest', extent=extent, aspect='auto',
              vmin=0.2, vmax=5.2)
    ax.plot(fiberap[numpy.argmax(hdu['SNR'].data[0], axis=0)], seeing, color='C1')
    coo = numpy.unravel_index(numpy.argmax(hdu['SNR'].data[0]), hdu['SNR'].data[0].shape)
    ax.scatter(Y[coo], X[coo], marker='x', s=100, lw=1, color='k')
#    ax.text(-0.1, 0.5, r'Fiber Diameter [arcsec]', ha='center', va='center',
#            transform=ax.transAxes, rotation='vertical')
    ax.text(0.5, 1.1, r'Point Source', ha='center', va='center', transform=ax.transAxes)

    ax = fig.add_axes([0.39, 0.73, 0.28, 0.21])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:])
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.imshow(hdu['SNR'].data[1].T,
              origin='lower', interpolation='nearest', extent=extent, aspect='auto',
              vmin=0.2, vmax=5.2)
    ax.plot(fiberap[numpy.argmax(hdu['SNR'].data[1], axis=0)], seeing, color='C1')
    coo = numpy.unravel_index(numpy.argmax(hdu['SNR'].data[1]), hdu['SNR'].data[1].shape)
    ax.scatter(Y[coo], X[coo], marker='x', s=100, lw=1, color='k')
    ax.text(0.5, 1.1, r'Sersic: $R_e=0.4$", $n=1$',
            ha='center', va='center', transform=ax.transAxes)

    ax = fig.add_axes([0.68, 0.73, 0.28, 0.21])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:])
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.imshow(hdu['SNR'].data[2].T,
              origin='lower', interpolation='nearest', extent=extent, aspect='auto',
              vmin=0.2, vmax=5.2)
    ax.plot(fiberap[numpy.argmax(hdu['SNR'].data[2], axis=0)], seeing, color='C1')
    coo = numpy.unravel_index(numpy.argmax(hdu['SNR'].data[2]), hdu['SNR'].data[1].shape)
    ax.scatter(Y[coo], X[coo], marker='x', s=100, lw=1, color='k')
    ax.text(1.07, 0.5, r'S/N', ha='center', va='center',
            transform=ax.transAxes, rotation='vertical')
    ax.text(0.5, 1.1, r'Sersic: $R_e=0.4$", $n=4$',
            ha='center', va='center', transform=ax.transAxes)



    ax = fig.add_axes([0.1, 0.51, 0.28, 0.21])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:])
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    img = numpy.square(hdu['SNR'].data[0])*frac[None,:]
    ax.imshow(img.T,
              origin='lower', interpolation='nearest', extent=extent, aspect='auto',
              vmin=-0.01, vmax=0.06)
    ax.plot(fiberap[numpy.argmax(img, axis=0)][:-1], seeing[:-1], color='C1')
    coo = numpy.unravel_index(numpy.argmax(img), img.shape)
    ax.scatter(Y[coo], X[coo], marker='x', s=100, lw=1, color='k')
    ax.text(-0.23, 0.5, r'Seeing, $\theta$ [arcsec]', ha='center', va='center',
            transform=ax.transAxes, rotation='vertical')

    ax = fig.add_axes([0.39, 0.51, 0.28, 0.21])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:])
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    img = numpy.square(hdu['SNR'].data[1])*frac[None,:]
    ax.imshow(img.T,
              origin='lower', interpolation='nearest', extent=extent, aspect='auto',
              vmin=-0.01, vmax=0.06)
    ax.plot(fiberap[numpy.argmax(img, axis=0)][:-1], seeing[:-1], color='C1')
    coo = numpy.unravel_index(numpy.argmax(img), img.shape)
    ax.scatter(Y[coo], X[coo], marker='x', s=100, lw=1, color='k')

    ax = fig.add_axes([0.68, 0.51, 0.28, 0.21])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:])
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    img = numpy.square(hdu['SNR'].data[2])*frac[None,:]
    ax.imshow(img.T,
              origin='lower', interpolation='nearest', extent=extent, aspect='auto',
              vmin=-0.01, vmax=0.06)
    ax.plot(fiberap[numpy.argmax(img, axis=0)][:-1], seeing[:-1], color='C1')
    coo = numpy.unravel_index(numpy.argmax(img), img.shape)
    ax.scatter(Y[coo], X[coo], marker='x', s=100, lw=1, color='k')
    ax.text(1.07, 0.5, r'(S/N)$^2\ P(\theta)$', ha='center', va='center',
            transform=ax.transAxes, rotation='vertical')



    ax = fig.add_axes([0.1, 0.29, 0.28, 0.21])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:])
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    img = numpy.square(hdu['SNR'].data[0])*frac[None,:]/fiberap[:,None]
    ax.imshow(img.T,
              origin='lower', interpolation='nearest', extent=extent, aspect='auto',
              vmin=-0.01, vmax=0.10)
    ax.plot(fiberap[numpy.argmax(img, axis=0)][:-1], seeing[:-1], color='C1')
    coo = numpy.unravel_index(numpy.argmax(img), img.shape)
    ax.scatter(Y[coo], X[coo], marker='x', s=100, lw=1, color='k')
#    ax.text(-0.1, 0.5, r'Fiber Diameter [arcsec]', ha='center', va='center',
#            transform=ax.transAxes, rotation='vertical')
#    ax.text(0.5, -0.1, r'Seeing [arcsec]', ha='center', va='center', transform=ax.transAxes)

    ax = fig.add_axes([0.39, 0.29, 0.28, 0.21])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:])
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    img = numpy.square(hdu['SNR'].data[1])*frac[None,:]/fiberap[:,None]
    ax.imshow(img.T,
              origin='lower', interpolation='nearest', extent=extent, aspect='auto',
              vmin=-0.01, vmax=0.10)
    ax.plot(fiberap[numpy.argmax(img, axis=0)][:-1], seeing[:-1], color='C1')
    coo = numpy.unravel_index(numpy.argmax(img), img.shape)
    ax.scatter(Y[coo], X[coo], marker='x', s=100, lw=1, color='k')

    ax = fig.add_axes([0.68, 0.29, 0.28, 0.21])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:])
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    img = numpy.square(hdu['SNR'].data[2])*frac[None,:]/fiberap[:,None]
    ax.imshow(img.T,
              origin='lower', interpolation='nearest', extent=extent, aspect='auto',
              vmin=-0.01, vmax=0.10)
    ax.plot(fiberap[numpy.argmax(img, axis=0)][:-1], seeing[:-1], color='C1')
    coo = numpy.unravel_index(numpy.argmax(img), img.shape)
    ax.scatter(Y[coo], X[coo], marker='x', s=100, lw=1, color='k')
#    ax.text(0.5, -0.1, r'Seeing [arcsec]', ha='center', va='center', transform=ax.transAxes)
    ax.text(1.07, 0.5, r'(S/N)$^2\ P(\theta)\ D_f^{-1}$', ha='center', va='center',
            transform=ax.transAxes, rotation='vertical')



    ilim = [0.009, 0.065]

    ax = fig.add_axes([0.1, 0.12, 0.28, 0.16])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(ilim)
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    img = [hdu['SNR'].data[0], numpy.square(hdu['SNR'].data[0])*frac[None,:],
           numpy.square(hdu['SNR'].data[0])*frac[None,:]/fiberap[:,None]]
    for i,_img in enumerate(img):
        integ = numpy.sum(_img, axis=1)
        integ /= numpy.sum(integ)
        ax.plot(fiberap, integ, color='C{0}'.format(i))
        ax.axvline(x=fiberap[numpy.argmax(integ)], color='C{0}'.format(i), lw=0.5)
    ax.text(-0.23, 0.5, r'Mean Over $\theta$', ha='center', va='center',
            transform=ax.transAxes, rotation='vertical')
#    ax.text(0.5, -0.1, r'Seeing [arcsec]', ha='center', va='center', transform=ax.transAxes)

    ax = fig.add_axes([0.39, 0.12, 0.28, 0.16])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(ilim)
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    img = [hdu['SNR'].data[1], numpy.square(hdu['SNR'].data[1])*frac[None,:],
           numpy.square(hdu['SNR'].data[1])*frac[None,:]/fiberap[:,None]]
    for i,_img in enumerate(img):
        integ = numpy.sum(_img, axis=1)
        integ /= numpy.sum(integ)
        ax.plot(fiberap, integ, color='C{0}'.format(i))
        ax.axvline(x=fiberap[numpy.argmax(integ)], color='C{0}'.format(i), lw=0.5)
    ax.text(0.5, -0.3, r'Fiber Diameter, $D_f$ [arcsec]', ha='center', va='center',
            transform=ax.transAxes)
#    ax.text(0.5, -0.15, r'Seeing, $\theta$ [arcsec]', ha='center', va='center', transform=ax.transAxes)

    ax = fig.add_axes([0.68, 0.12, 0.28, 0.16])
    ax.minorticks_on()
    ax.set_xlim(extent[0:2])
    ax.set_ylim(ilim)
    ax.tick_params(which='major', length=10, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=5, direction='in', top=True, right=True)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    img = [hdu['SNR'].data[2], numpy.square(hdu['SNR'].data[2])*frac[None,:],
           numpy.square(hdu['SNR'].data[2])*frac[None,:]/fiberap[:,None]]
    for i,_img in enumerate(img):
        integ = numpy.sum(_img, axis=1)
        integ /= numpy.sum(integ)
        ax.plot(fiberap, integ, color='C{0}'.format(i))
        ax.axvline(x=fiberap[numpy.argmax(integ)], color='C{0}'.format(i), lw=0.5)
#    ax.text(0.5, -0.1, r'Seeing [arcsec]', ha='center', va='center', transform=ax.transAxes)
#    ax.text(1.1, 0.5, r'(S/N)$^2\ P(\theta)\ D_f^{-1}$', ha='center', va='center',
#            transform=ax.transAxes, rotation='vertical')

    fig.canvas.print_figure('optimal_aperture_wgt.pdf', bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)

    exit()

    pyplot.imshow(numpy.square(snr)*frac[None,:], origin='lower', interpolation='nearest')
    pyplot.show()

    pyplot.imshow(numpy.square(snr)*frac[None,:]/fiberap[:,None], origin='lower', interpolation='nearest')
    pyplot.show()



    exit()


    mag = 24.
    ofile = 'optimal_aperture_{0:.1f}.fits'.format(mag)
    if not os.path.isfile(ofile):
        build_data(mag, ofile)

    make_plots(ofile)

