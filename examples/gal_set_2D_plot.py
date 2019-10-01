
import numpy
from matplotlib import pyplot, colors

from astropy.io import fits

# H-alpha:
#   9800:10800,7900:9000
# H-beta:
#   2250:4100,100:1650

def main():
    hdu = fits.open('gal_set_red.fits')

    data = hdu[0].data[3001:15501,61:10701]+1e-5
    data_aspect = data.shape[0]/data.shape[1]

    ha_box = numpy.array([[9800, 6800],
                          [9800, 9000],
                          [10800, 9000],
                          [10800, 6800],
                          [9800, 6800]])
    
    dataha = data[9800:10800,6800:9000]
    subhaw = 0.18 #8*dataha.shape[1]/data.shape[1]
    dataha_aspect = dataha.shape[0]/dataha.shape[1]

    hb_box = numpy.array([[2250, 100],
                          [2250, 1650],
                          [3600, 1650],
                          [3600, 100],
                          [2250, 100]])
    
    datahb = data[2250:3600,100:1650]
    subhbw = 0.26 #8*dataha.shape[1]/data.shape[1]
    datahb_aspect = datahb.shape[0]/datahb.shape[1]

    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(1.5*w,1.5*h))

    ax = fig.add_axes([(1-0.8/data_aspect)/2, 0.1, 0.8/data_aspect, 0.8])
    ax.imshow(data, origin='lower', interpolation='nearest',
              norm=colors.LogNorm(vmin=1, vmax=500), cmap='inferno', aspect='auto', zorder=0)
    ax.plot(ha_box[:,1], ha_box[:,0], alpha=0.7, color='w', zorder=1, lw=2)
    ax.plot(hb_box[:,1], hb_box[:,0], alpha=0.7, color='w', zorder=1, lw=2)
    ax.text(0.5, -0.06, 'Spatial Coordinate (pixels)', ha='center', va='center',
            transform=ax.transAxes)
    ax.text(-0.13, 0.5, 'Spectral Coordinate (pixels)', ha='center', va='center',
            transform=ax.transAxes, rotation='vertical')
    ax.text(0.5, 1.02, 'Simulated WFOS Red Detector', ha='center', va='center',
            transform=ax.transAxes)

    ax = fig.add_axes([0.23, 0.6, subhaw/dataha_aspect, subhaw])
    ax.imshow(dataha, origin='lower', interpolation='nearest',
              norm=colors.LogNorm(vmin=1, vmax=500), cmap='inferno', aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = fig.add_axes([0.25, 0.15, subhbw/datahb_aspect, subhbw])
    ax.imshow(datahb, origin='lower', interpolation='nearest',
              norm=colors.LogNorm(vmin=1, vmax=500), cmap='inferno', aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
   
    fig.canvas.print_figure('simulated_red_camera.png', bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)

if __name__ == '__main__':
    main()


