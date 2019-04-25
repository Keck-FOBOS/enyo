import os
import warnings

import numpy
from scipy import interpolate

from matplotlib import pyplot, ticker, rc, colors, cm, colorbar, image, patches

from enyo.etc import source, telescopes, aperture, spectrum, efficiency

def test_mag():
    wave = numpy.linspace(3000., 10000., num=7001)
    spec = spectrum.ABReferenceSpectrum(wave)
    g = efficiency.FilterResponse()
    print(spec.magnitude(g))

    spec.rescale_magnitude(g, 10)
    print(spec.magnitude(g))

#    pyplot.plot(spec.wave, spec.flux)
#    pyplot.plot(spec.wave, spec.flux)
#    pyplot.yscale('log')
#    pyplot.show()

    pyplot.plot(spec.wave, spec.photon_flux())
    pyplot.show()
    print(spec.photon_flux(band=g))

def fiddles_data_table(seeing, exptime, airmass, dmag_sky, data_file):
    mag = numpy.linspace(15, 22, 15, dtype=float)[::-1]
    apf_snr_pix = numpy.zeros_like(mag)
    apf_snr = numpy.zeros_like(mag)
    apf_saturated = numpy.zeros_like(mag, dtype=bool)

    keck_snr_pix = numpy.zeros_like(mag)
    keck_snr = numpy.zeros_like(mag)
    keck_saturated = numpy.zeros_like(mag, dtype=bool)

    keck = telescopes.KeckTelescope()
    apf = telescopes.APFTelescope()

    for i in range(mag.size):
        apf_snr_pix[i], apf_snr[i], apf_saturated[i], sky_mag \
                    = fiddles_snr(mag[i], apf, seeing, exptime, airmass=airmass,
                                  dmag_sky=dmag_sky)
        keck_snr_pix[i], keck_snr[i], keck_saturated[i], sky_mag \
                    = fiddles_snr(mag[i], keck, seeing, exptime, airmass=airmass,
                                  dmag_sky=dmag_sky)

    numpy.savetxt(data_file, numpy.transpose([mag, apf_snr_pix, apf_snr, apf_saturated.astype(int),
                                          keck_snr_pix, keck_snr, keck_saturated.astype(int)]),
                  fmt=['%5.1f', '%14.7e', '%14.7e', '%2d', '%14.7e', '%14.7e', '%2d'],
                  header='Sky surface brightness: {0:.2f}'.format(sky_mag) +
                         'Exposure time: {0:.1f}\n'.format(exptime) +
                         'Airmass: {0:.1f}\n'.format(airmass) +
                         'Seeing: {0:.1f}\n'.format(seeing) +
                         '{0:>3s} {1:^32} {2:^32}\n'.format('', 'APF', 'Keck') +
                         '{0:>3s} {1:^32} {2:^32}\n'.format('', '-'*32, '-'*32) +
                         '{0:>3s} {1:>14s} {2:>14s} {3:>2s} {4:>14s} {5:>14s} {6:>2s}'.format('Mg',
                            'SNRp', 'SNRt', 'S', 'SNRp', 'SNRt', 'S'))

def main(force):

#    telescope = telescopes.KeckTelescope()
#    print(fiddles_snr(19, telescope, 0.8, 60))
#    exit()

    seeing = 0.8        # arcsec FWHM
    airmass = 2         # maximum airmass

    exptime = 60        # sec
    dmag_sky = None     # Dark time
    ofile = os.path.join(os.environ['ENYO_DIR'], 'fiddles_etc_dark.out')
    if not os.path.isfile(ofile) or force:
        fiddles_data_table(seeing, exptime, airmass, dmag_sky, ofile)
    oplot = os.path.join(os.environ['ENYO_DIR'], 'fiddles_etc_dark.pdf')
#    if not os.path.isfile(oplot) or force:
    fiddles_snr_plot(seeing, exptime, airmass, dmag_sky, ofile, plot_file=oplot)

    exptime = 60        # sec
    dmag_sky = -3.0     # Bright time...
    ofile = os.path.join(os.environ['ENYO_DIR'], 'fiddles_etc_bright.out')
    if not os.path.isfile(ofile) or force:
        fiddles_data_table(seeing, exptime, airmass, dmag_sky, ofile)
    oplot = os.path.join(os.environ['ENYO_DIR'], 'fiddles_etc_bright.pdf')
#    if not os.path.isfile(oplot) or force:
    fiddles_snr_plot(seeing, exptime, airmass, dmag_sky, ofile, plot_file=oplot)


def  fiddles_snr_plot(seeing, exptime, airmass, dmag_sky, data_file, plot_file=None):

    db = numpy.genfromtxt(data_file)
    mag = db[:,0]
    apf_snr_pix = db[:,1]
    apf_snr = db[:,2]
    apf_saturated = db[:,3].astype(bool)
    keck_snr_pix = db[:,4]
    keck_snr = db[:,5]
    keck_saturated = db[:,6].astype(bool)

    # TODO: Read this from the file header
    g = efficiency.FilterResponse()
    sky = spectrum.MaunakeaSkySpectrum()
    sky_mag = sky.magnitude(g)
    if dmag_sky is not None:
        sky_mag += dmag_sky

#    best_apf_mag = interpolate.interp1d(apf_snr_pix, mag)(10.)
#    print(best_apf_mag)
#    best_keck_mag = interpolate.interp1d(keck_snr_pix, mag)(10.)
#    print(best_keck_mag)

    font = { 'size' : 16 }
    rc('font', **font)

    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))

    ax = fig.add_axes([0.15, 0.15, 0.69, 0.69], facecolor='0.98')
    ax.minorticks_on()
    ax.tick_params(which='major', length=6, direction='in')
    ax.tick_params(which='minor', length=3, direction='in')
    ax.grid(True, which='major', color='0.9', zorder=0, linestyle='-')

    ax.plot(mag[numpy.invert(apf_saturated)], apf_snr_pix[numpy.invert(apf_saturated)],
            color='C0', zorder=2, lw=0.5)
    ax.scatter(mag[numpy.invert(apf_saturated)], apf_snr_pix[numpy.invert(apf_saturated)],
               marker='.', lw=0, color='C0', s=100, zorder=2)
    if numpy.any(apf_saturated):
        ax.scatter(mag[apf_saturated], apf_snr_pix[apf_saturated],
                   marker='x', lw=1, color='C3', s=50, zorder=2)

    ax.plot(mag[numpy.invert(keck_saturated)], keck_snr_pix[numpy.invert(keck_saturated)],
            color='C2', zorder=2, lw=0.5)
    ax.scatter(mag[numpy.invert(keck_saturated)], keck_snr_pix[numpy.invert(keck_saturated)],
               marker='.', lw=0, color='C2', s=100, zorder=2)
    if numpy.any(keck_saturated):
        ax.scatter(mag[keck_saturated], keck_snr_pix[keck_saturated],
                   marker='x', lw=1, color='C3', s=50, zorder=2)

    ax.axhline(10., color='k', linestyle='--')

    ax.set_ylim([0.01, 300])
    ax.set_yscale('log')
    logformatter = ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(
                                            int(numpy.maximum(-numpy.log10(y),0)))).format(y))
    ax.yaxis.set_major_formatter(logformatter)
    ax.text(-0.13, 0.5, r'S/N (pixel$^{-1}$)', ha='center', va='center',
            rotation='vertical', transform=ax.transAxes)
    ax.text(0.5, -0.1, r'$m_g$ (AB Mag)', ha='center', va='center',
            transform=ax.transAxes)
    ax.text(0.05, 0.35, 'APF', ha='left', va='center', transform=ax.transAxes, color='C0')
    ax.text(0.05, 0.30, 'Keck', ha='left', va='center', transform=ax.transAxes, color='C2')
    ax.text(0.05, 0.25, 'Saturated', ha='left', va='center', transform=ax.transAxes, color='C3')
    ax.text(0.05, 0.20, r'Sky $\mu_g$ = {0:.2f} AB mag / arcsec$^2$'.format(sky_mag),
            ha='left', va='center', transform=ax.transAxes, color='k')
    ax.text(0.05, 0.15, 'ExpTime = {0} s'.format(exptime), ha='left', va='center',
            transform=ax.transAxes, color='k')
    ax.text(0.05, 0.05, 'Seeing = {0} arcsec FWHM'.format(seeing), ha='left', va='center',
            transform=ax.transAxes, color='k')
    ax.text(0.05, 0.10, 'Airmass = {0}'.format(airmass), ha='left', va='center',
            transform=ax.transAxes, color='k')

    if plot_file is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(plot_file, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)


def fiddles_model(mag, telescope, seeing, exposure_time, airmass=1.0, fiddles_fratio=3.,
                 fiber_diameter=0.15, dmag_sky=None):

    # Use a reference spectrum that's constant; units are 1e-17 erg/s/cm^2/angstrom
    wave = numpy.linspace(3000., 10000., num=7001)
    spec = spectrum.ABReferenceSpectrum(wave)
    g = efficiency.FilterResponse()

    # Rescale to a specified magnitude
    spec.rescale_magnitude(g, mag)
    print('Star Magnitude (AB mag): {0:.2f}'.format(spec.magnitude(g)))

    # Sky Spectrum; units are 1e-17 erg/s/cm^2/angstrom/arcsec^2
    sky = spectrum.MaunakeaSkySpectrum()
    sky_mag = sky.magnitude(g)
    print('Sky Surface Brightness (AB mag/arcsec^2): {0:.2f}'.format(sky_mag))
    if dmag_sky is not None:
        sky_mag += dmag_sky
        sky.rescale_magnitude(g, sky_mag)

    # Fiddles detector
    fiddles_pixel_size = 7.4e-3 # mm
    fiddles_fullwell = 18133 # Electrons (NOT ADU!)
    fiddles_platescale = telescope.platescale*fiddles_fratio/telescope.fratio
    qe_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'efficiency', 'detectors',
                           'thor_labs_monochrome_qe.db')
    fiddles_det = efficiency.Detector(pixelscale=fiddles_pixel_size/fiddles_platescale,
                                      rn=6.45, dark=5.,
                                      qe=efficiency.Efficiency.from_file(qe_file, wave_units='nm'))
    print('FIDDLES pixelscale (arcsec/pixel): {0:.3f}'.format(fiddles_det.pixelscale))

    # On-sky fiber diameter
    on_sky_fiber_diameter = fiber_diameter/fiddles_platescale   # arcsec
    print('On-sky fiber diameter (arcsec): {0:.2f}'.format(on_sky_fiber_diameter))

    # Size for the fiber image
    upsample = 1 if fiddles_det.pixelscale < seeing/5 else int(fiddles_det.pixelscale/(seeing/5))+1
    print('Upsampling: {0}'.format(upsample))
    sampling = fiddles_det.pixelscale/upsample
    print('Pixel sampling of maps: {0:.3f} (arcsec/pixel)'.format(sampling))
    size = (int(numpy.ceil(on_sky_fiber_diameter*1.2/sampling))//upsample + 1)*upsample*sampling
    print('Square map size : {0:.3f} (arcsec)'.format(size))

    # Use a point source with unity integral (integral of mapped
    # profile will not necessarily be unity)
    star = source.OnSkySource(seeing, 1.0, sampling=sampling, size=size)
    print('Fraction of star flux in mapped image: {0:.3f}'.format(numpy.sum(star.data)
                                                                  * numpy.square(sampling)))
#    pyplot.imshow(star.data, origin='lower', interpolation='nearest')
#    pyplot.show()

    # Define fiber aperture
    fiber = aperture.FiberAperture(0., 0., on_sky_fiber_diameter, resolution=100)
    # Generate its response function
    fiber_response = fiber.response(star.x, star.y)
    print('Fiber area (mapped): {0:.2f} (arcsec^2)'.format(numpy.sum(fiber_response)
                                                           * numpy.square(star.sampling)))
    print('Fiber area (nominal): {0:.2f} (arcsec^2)'.format(fiber.area))

    in_fiber = numpy.sum(star.data*fiber_response)*numpy.square(star.sampling)
    print('Aperture loss: {0:.3f}'.format(in_fiber))

    # This would assume that the fiber does NOT scramble the light...
#    fiber_img = star.data*fiber_response

    # Instead scale the fiber response profile to produce a uniformly
    # scrambled fiber output beam
    scale = numpy.sum(star.data*fiber_response)/numpy.sum(fiber_response)
    source_fiber_img = scale*fiber_response

    # Down-sample to the detector size
    if upsample > 1:
        source_fiber_img = numpy.add.reduceat(source_fiber_img,
                                              numpy.arange(0,int(size/sampling),upsample),
                                              axis=0)/upsample
        source_fiber_img = numpy.add.reduceat(source_fiber_img,
                                              numpy.arange(0,int(size/sampling),upsample),
                                              axis=1)/upsample

        fiber_response = numpy.add.reduceat(fiber_response,
                                            numpy.arange(0,int(size/sampling),upsample),
                                            axis=0)/upsample
        fiber_response = numpy.add.reduceat(fiber_response,
                                            numpy.arange(0,int(size/sampling),upsample),
                                            axis=1)/upsample

    # Scale the spectra to be erg/angstrom (per arcsec^2 for sky)
    print('Exposure time: {0:.1f} (s)'.format(exposure_time))
    spec.rescale(telescope.area * exposure_time)
    sky.rescale(telescope.area * exposure_time)

    # Convert them to photons/angstrom (per arcsec^2 for sky)
    spec.photon_flux()
    sky.photon_flux()

    # Atmospheric Extinction
    atm = efficiency.AtmosphericThroughput(airmass=airmass)

    # Fiber throughput (10-m polymicro fiber run)
    fiber_throughput = efficiency.FiberThroughput()

    # TODO: assumes no fiber FRD or coupling losses

    # Integrate spectrally to get the total number of electrons at the
    # detector accounting for the g-band filter and the detector qe.
    flux = numpy.sum(spec.flux * atm(spec.wave) * fiber_throughput(spec.wave) * g(spec.wave)
                      * fiddles_det.qe(spec.wave) * spec.wavelength_step())
    print('Object flux: {0:.3f} (electrons)'.format(flux))
    sky_flux = numpy.sum(sky.flux * fiber_throughput(sky.wave) * g(sky.wave)
                         * fiddles_det.qe(sky.wave) * sky.wavelength_step())
    print('Sky flux: {0:.3f} (electrons/arcsec^2)'.format(sky_flux))
    print('Sky flux: {0:.3f} (electrons)'.format(sky_flux*fiber.area))
    
    # Scale the source image by the expected flux (source_fiber_img was
    # constructed assuming unity flux)
    fiber_obj_img = source_fiber_img * flux
    # Scale the fiber response function by the sky surface brightness
    # (the integral of the sky image is then just the fiber area times
    # the uniform sky surface brightness)
    fiber_sky_img = fiber_response * sky_flux

    # Integrate spatially over the pixel size to get the number of
    # electrons in each pixel
    fiber_obj_img *= numpy.square(upsample*sampling)
    fiber_sky_img *= numpy.square(upsample*sampling)

    return fiber_obj_img, fiber_sky_img, fiddles_det, sky_mag


def fiddles_snr(mag, telescope, seeing, exposure_time, airmass=1.0, fiddles_fratio=3.,
                fiber_diameter=0.15, dmag_sky=None):

    fiddles_fullwell = 18133 # Electrons (NOT ADU!)
    fiber_obj_img, fiber_sky_img, fiddles_det, sky_mag \
            = fiddles_model(mag, telescope, seeing, exposure_time, airmass=airmass,
                            fiddles_fratio=fiddles_fratio, fiber_diameter=fiber_diameter,
                            dmag_sky=dmag_sky)

    # Check if the image is saturated in any pixel
    tot_img = fiber_obj_img + fiber_sky_img + fiddles_det.dark*exposure_time
    print('Maximum number of electrons per pixel: {0:.2f}'.format(numpy.max(tot_img)))

    saturated = numpy.any(tot_img > fiddles_fullwell)
    if saturated:
        warnings.warn('FIDDLES will be saturated!')

    # Compute the variance in each pixel of the object image and the sky image
    fiber_sky_var = fiber_sky_img + fiddles_det.rn**2 + fiddles_det.dark*exposure_time
    fiber_obj_var = fiber_obj_img + fiber_sky_var

    # Compute the S/N image of the output near field *difference* image
    snr_img = fiber_obj_img/numpy.sqrt(fiber_obj_var + fiber_sky_var)
    n = snr_img.shape[0]
    print('In-fiber S/N per pixel: {0:.3f}'.format(snr_img[n//2,n//2]))

    # Compute the S/N of the extracted flux (assuming perfect extraction)
    snr = numpy.sum(fiber_obj_img) / numpy.sqrt(numpy.sum(fiber_obj_var + fiber_sky_var))
    print('Expected total S/N: {0:.2f}'.format(snr))

    return snr_img[n//2,n//2], snr, saturated, sky_mag

#    pyplot.imshow(snr_img, origin='lower', interpolation='nearest')
#    pyplot.colorbar()
#    pyplot.show()


def fiddles_realization(mag, telescope, seeing, exposure_time, airmass=1.0, fiddles_fratio=3.,
                        fiber_diameter=0.15, dmag_sky=None):

    fiddles_fullwell = 18133 # Electrons (NOT ADU!)

    # Bias level?

    # Build the model
    fiber_obj_img, fiber_sky_img, fiddles_det, sky_mag \
            = fiddles_model(mag, telescope, seeing, exposure_time, airmass=airmass,
                            fiddles_fratio=fiddles_fratio, fiber_diameter=fiber_diameter,
                            dmag_sky=dmag_sky)

    # Get the noise for the on-source image
    obj_noise_img = numpy.random.poisson(lam=fiber_obj_img)
    sky_noise_img = numpy.random.poisson(lam=fiber_sky_img)
    dark_noise_img = numpy.random.poisson(
                            lam=numpy.full_like(fiber_sky_img, fiddles_det.dark*exposure_time))
    read_noise_img = numpy.random.normal(scale=fiddles_det.rn,
                                         size=obj_noise_img.size).reshape(obj_noise_img.shape)

    # Get the on-source realization
    on_img = fiber_obj_img + fiber_sky_img + fiddles_det.dark*exposure_time
    on_img += obj_noise_img + sky_noise_img + dark_noise_img + read_noise_img

    # Get the noise for the off-source image
    sky_noise_img = numpy.random.poisson(lam=fiber_sky_img)
    dark_noise_img = numpy.random.poisson(
                            lam=numpy.full_like(fiber_sky_img, fiddles_det.dark*exposure_time))
    read_noise_img = numpy.random.normal(scale=fiddles_det.rn,
                                         size=obj_noise_img.size).reshape(obj_noise_img.shape)

    off_img = fiber_sky_img + fiddles_det.dark*exposure_time
    off_img += sky_noise_img + dark_noise_img + read_noise_img

    return on_img, off_img


def fiddles_realization_plot(plot_file=None):
    telescope = telescopes.KeckTelescope()
    on, off = fiddles_realization(19., telescope, 0.8, 60, airmass=2., dmag_sky=-3)

    vmin=-100
    vmax=3500

    font = { 'size' : 10 }
    rc('font', **font)

    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))

    ax = fig.add_axes([0.07, 0.35, 0.3, 0.3])
    cax = fig.add_axes([0.07, 0.72, 0.2, 0.01])
    ax.minorticks_on()
    ax.tick_params(which='major', length=6, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=3, direction='in', top=True, right=True)
#    ax.grid(True, which='major', color='0.9', zorder=0, linestyle='-')

    im = ax.imshow(on, origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
    pyplot.colorbar(im, cax=cax, orientation='horizontal')
    cax.text(1.1, 0.5, r'Counts (e$-$)', ha='left', va='center', transform=cax.transAxes)

    ax.text(-0.15, 0.5, r'Y (pixels)', ha='center', va='center',
             rotation='vertical', transform=ax.transAxes)
    ax.text(0.5, -0.14, r'X (pixels)', ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 1.05, r'On Source', ha='center', va='center', transform=ax.transAxes)

    ax = fig.add_axes([0.37, 0.35, 0.3, 0.3])
    ax.minorticks_on()
    ax.tick_params(which='major', length=6, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=3, direction='in', top=True, right=True)
#    ax.grid(True, which='major', color='0.9', zorder=0, linestyle='-')
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    ax.imshow(off, origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)

    ax.text(0.5, -0.14, r'X (pixels)', ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 1.05, r'Off Source', ha='center', va='center', transform=ax.transAxes)

    ax = fig.add_axes([0.67, 0.35, 0.3, 0.3])
    ax.minorticks_on()
    ax.tick_params(which='major', length=6, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=3, direction='in', top=True, right=True)
#    ax.grid(True, which='major', color='0.9', zorder=0, linestyle='-')
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    ax.imshow(on-off, origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)

    ax.text(0.5, -0.14, r'X (pixels)', ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 1.05, r'On-Off', ha='center', va='center', transform=ax.transAxes)

    if plot_file is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(plot_file, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)


def spec_eff_plot(plot_file=None):

    mag = 19.
    airmass = 2.
    dmag_sky = -3.

    # Spectra; units are 1e-17 erg/s/cm^2/angstrom/arcsec^2
    wave = numpy.linspace(3000., 10000., num=7001)
    spec = spectrum.ABReferenceSpectrum(wave)
    g = efficiency.FilterResponse()
    spec.rescale_magnitude(g, mag)
    sky = spectrum.MaunakeaSkySpectrum()
    sky_mag = sky.magnitude(g)
    if dmag_sky is not None:
        sky_mag += dmag_sky
        sky.rescale_magnitude(g, sky_mag)

    # Efficiencies
    atm = efficiency.AtmosphericThroughput(airmass=airmass)
    fiber_throughput = efficiency.FiberThroughput()

    qe_file = os.path.join(os.environ['ENYO_DIR'], 'data', 'efficiency', 'detectors',
                           'thor_labs_monochrome_qe.db')
    fiddles_qe = efficiency.Efficiency.from_file(qe_file, wave_units='nm')

    xlim = [3700, 5600]

    font = { 'size' : 10 }
    rc('font', **font)

    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))

    ax = fig.add_axes([0.1, 0.5, 0.85, 0.2])
    ax.minorticks_on()
    ax.tick_params(which='major', length=6, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=3, direction='in', top=True, right=True)
    ax.grid(True, which='major', color='0.9', zorder=0, linestyle='-')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())

    ax.set_xlim(xlim)
    ax.set_ylim([3, 35])

    ax.plot(spec.wave, spec.flux, color='k', zorder=3)
    ax.plot(sky.wave, sky.flux, color='C3', zorder=3, lw=0.8)

    ax.text(-0.08, 0.5, r'$F_\lambda$ (erg/s/cm$^2$/${\rm \AA}$)', ha='center', va='center',
            rotation='vertical', transform=ax.transAxes)
    ax.text(-0.05, 0.5, r'$\mathcal{I}_\lambda$ (erg/s/cm$^2$/${\rm \AA}$/")', ha='center', va='center',
            rotation='vertical', transform=ax.transAxes, color='C3')
    ax.text(0.05, 0.55, 'Source', ha='left', va='center', transform=ax.transAxes)
    ax.text(0.2, 0.1, 'Sky', color='C3', ha='left', va='center', transform=ax.transAxes)

    ax = fig.add_axes([0.1, 0.3, 0.85, 0.2])
    ax.minorticks_on()
    ax.tick_params(which='major', length=6, direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=3, direction='in', top=True, right=True)
    ax.grid(True, which='major', color='0.9', zorder=0, linestyle='-')

    ax.set_xlim(xlim)
    ax.set_ylim([0, 1.1])

    ax.plot(atm.wave, atm.eta, color='C3', zorder=3)
    ax.plot(fiber_throughput.wave, fiber_throughput.eta, color='C0', zorder=3)
    ax.plot(fiddles_qe.wave, fiddles_qe.eta, color='C1', zorder=3)
    ax.plot(g.wave, g.eta, color='k', zorder=3)

    ax.text(0.5, -0.2, r'Wavelength (${\rm \AA}$)', ha='center', va='center',
            transform=ax.transAxes)
    ax.text(-0.08, 0.5, r'$\eta$', ha='center', va='center', rotation='vertical',
            transform=ax.transAxes)
    ax.text(0.01, 0.87, 'Fiber', color='C0', ha='left', va='center', transform=ax.transAxes)
    ax.text(0.01, 0.57, 'Sky', color='C3', ha='left', va='center', transform=ax.transAxes)
    ax.text(0.01, 0.4, 'Detector', color='C1', ha='left', va='center', transform=ax.transAxes)
    ax.text(0.01, 0.1, r'$g$ Filter', color='k', ha='left', va='center', transform=ax.transAxes)

    if plot_file is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(plot_file, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)


if __name__ == '__main__':

    spec_eff_plot('fiddles_etc_elements.pdf')

    plot_file = os.path.join(os.environ['ENYO_DIR'], 'fiddles_etc_realization.pdf')
    fiddles_realization_plot(plot_file=plot_file)

    main(False)
    exit()

    wave = numpy.linspace(3000., 10000., num=7001)
    spec = spectrum.ABReferenceSpectrum(wave)
    g = efficiency.FilterResponse()
#    print('Magnitude: {0:.2f}'.format(spec.magnitude(g)))

    spec.rescale_magnitude(g, 22.0)
    print('Magnitude: {0:.2f}'.format(spec.magnitude(g)))


    # Sky Spectrum; units are 1e-17 erg/s/cm^2/angstrom/arcsec^2
    sky = spectrum.MaunakeaSkySpectrum()
    print('Sky Magnitude: {0:.2f}'.format(sky.magnitude(g)))

    pyplot.plot(sky.wave, sky.flux)
    pyplot.plot(spec.wave, spec.flux)
    pyplot.plot(g.wave, g.eta)
    pyplot.show()

    #test_mag()