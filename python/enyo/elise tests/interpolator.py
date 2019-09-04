# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:32:57 2019

@author: sueel
"""
import enyo
import numpy as np
import os
import math
import warnings
import os.path
import matplotlib.pyplot as plt
from matplotlib import colors, cm, figure
from scipy import interpolate
from matplotlib import rc
from enyo.ARBTools.ARBInterp import tricubic

#external functions   
def max_thin_step(grid):
    maxlen = np.amax(grid.shape)
    if maxlen <= 3:
        return 1
    return int(np.amax(grid.shape)/2)//2


def iteration_steps(grid, max_steps=None):
    max_thin = max(max_thin_step(grid), 0 if max_steps is None else max_steps)
    iters = np.arange(max_thin)+1
    return np.append(iters, -iters[::-1])[:-1]


def grid_thin_indx(grid, step):

    if np.any(np.array(grid.shape) % 2 != 1):
        warnings.warn('Best if grid size is odd in all dimensions.')

    # Flattened index of every element
    indx = np.arange(np.prod(grid.shape)).reshape(grid.shape)

    # If step is -1, that means return the whole thing
    if step == -1:
        return indx.ravel()

    # Determine how to step and flip
    flip = step < 0
    _step = np.absolute(step)

    if _step > max_thin_step(grid):
        raise ValueError('Maximum thin step is {0}!.'.format(max_thin_step(grid)))

    # Ends and middle are always included
    bounding_sl = ()
    for i in range(grid.ndim):
        bounding_sl += (slice(None,None,max(grid.shape[i]//2,1)),)

    thin_indx = indx[bounding_sl]

    # If step is one, we're done
    if step == 1:
        return thin_indx.ravel()

    internal_sl = ()
    for i in range(grid.ndim):
        internal_sl += (slice(None,None,max(grid.shape[i]//(2*_step),1)),)

    if flip:
        keep = np.ones(grid.shape, dtype=bool)
        keep[internal_sl] = False
        return np.unique(np.append(thin_indx, indx[keep]))

    return np.unique(np.append(thin_indx, indx[internal_sl]))


def distance(out_c, y):
    #distance between truth and interp points to determine interp accuracy
    dist = []
    for i in range (out_c[:,0].size):
        distx = math.pow((out_c[i,0]-y[i,0]),2)
        disty = math.pow((out_c[i,1]-y[i,1]),2)
        dist = np.append(dist, (np.sqrt(distx+disty)))
    #dist = np.nanmedian(dist)
    return dist
   
    
def grabdata(db,indx):
     inxf = db[indx,0]
     inyf = db[indx,1]
     inlam = db[indx,2]
     inxc = db[indx,3]/0.015
     inyc = db[indx,4]/0.015
     raysthru = db[indx,5]
     return inxf,inyf,inlam,inxc,inyc,raysthru
 
    
#FP2D class & Red/Blue derived classes
class FocalPlane2Detector:
    #only work with data directly
    def __init__(self, inxf, inyf, inlam, inxc, inyc, raysthru):
        """
        init shoud:
            generate elements from file
            set up interpolator
        """    
        #f, c, & xi for NDInterp and transpose (.T)
        self.f = np.array([inxf.ravel(), inyf.ravel(), inlam.ravel()]).T
        
        self.c = np.array([ inxc.ravel(), inyc.ravel()]).T
        self.cpt = np.array([inxc.ravel(), inyc.ravel(), raysthru.ravel()]).T
        
        self.interpolator = None
        self.inverse = None
        
        self.interptri = None
        self.gridsize = self.f.shape[0]

    #make it just call self.interpolator, pass it xf yf lam to interp @, returns interpolated 
    def interp(self, outxf, outyf, outlam):
        if self.interpolator is None:    
            self.interpolator = interpolate.LinearNDInterpolator(self.f,self.c)
            
        xi = np.array([outxf.ravel(), outyf.ravel(), outlam.ravel()]).T

        return self.interpolator(xi)
    
    def invinterp(self, outxc, outyc, outyf):
        if self.inverse is None:    
            self.inverse = interpolate.LinearNDInterpolator(np.array([self.c[:,0], self.c[:,1], self.f[:,1]]).T, self.f[:,[0,2]])
            
        xi = np.array([outxc.ravel(), outyc.ravel(), outyf.ravel()]).T

        return self.inverse(xi)
    
    #interpolates percentage of rays thru
    def interpraysthru(self, outxf, outyf, outlam):
        if self.interpolator is None:    
            self.interpolator = interpolate.LinearNDInterpolator(self.f,self.cpt)
            
        xi = np.array([outxf.ravel(), outyf.ravel(), outlam.ravel()]).T

        return self.interpolator(xi)
      
#previously WFOSRedFocalPlane2Detector
class WFOSFocalPlane2Detector(FocalPlane2Detector):
    def __init__(self,whatfile,nx,ny,nl,step=None):
        modelfile = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', whatfile)
        self.db = np.genfromtxt(modelfile)
        
        nrows = nx*ny*nl
        row = np.arange(nrows).reshape(nx,ny,nl)
    
        # Get the rows to use in the interpolator
        if step is not None:
            thinned_rows = grid_thin_indx(row, step)
            self.use_row_for_grid = np.zeros(row.size, dtype=bool)
            self.use_row_for_grid[thinned_rows] = True
        else:
            self.use_row_for_grid = np.ones(row.size, dtype=bool) 
        
        self.use_row_for_grid &= (self.db[:,-1] > 0)
        
        inxf, inyf, inlam, self.inxc, self.inyc, self.raysthru = grabdata(self.db,self.use_row_for_grid)
        #adding %raysthru to c
        
        super(WFOSFocalPlane2Detector,self).__init__(inxf, inyf, inlam, self.inxc, self.inyc, self.raysthru)
        self.gridfrac = self.gridsize/self.db.shape[0]

   
def tracefig():
    """
    7.23.19
    function that generates/reads text files for random field points and interpolated points for 
    specified files (under filenames)
    also plots (and saves) interpolated points as well as random field points used
    """
    plt.close('all')
    nfield = 20
    lstep = 10
    fieldfile = '{0}pts.txt'.format(nfield)
    rc('font', size=14)
    
    if  os.path.isfile(fieldfile) is True:
        data = np.genfromtxt(fieldfile)
        random_x = data[:,0]
        random_y = data[:,1]
    else:
        random_x = np.random.uniform(-4.2,4.2,size=nfield)
        random_y = np.random.uniform(-1.5,1.5,size=nfield)
        np.savetxt(fieldfile, np.array([random_x,random_y]).T, fmt = ["%7.4f", "%7.4f"])
  
    plt.figure(figsize=(20,10))
    filenames = ['Blue_Low_Res_Spot_Data_9x9F_90W.txt', 'Blue_Low_Spot_Data_2020_150.txt', 'Blue_Low_Spot_Data_4040_300.txt','Red_Low_Res_Spot_Data_9x9F_90W.txt', 'Red_Low_Spot_Data_2020_150.txt', 'Red_Low_Spot_Data_4040_300.txt']
    ax = None
    
    for i in filenames:
        if i is 'Blue_Low_Res_Spot_Data_9x9F_90W.txt':
            res = 'blue low'
            xy = 9
            lamb = 93
            subp = 242
            outdat = 'outdatabluel.txt'
        elif i is 'Red_Low_Res_Spot_Data_9x9F_90W.txt':
            res = 'red low'
            xy = 9
            lamb = 93
            subp = 246
            outdat = 'outdataredl.txt'
        elif i is 'Blue_Low_Spot_Data_2020_150.txt':
            res = 'blue mid'
            xy = 20
            lamb = 150
            subp = 243
            outdat = 'outdatabluem.txt'
        elif i is 'Red_Low_Spot_Data_2020_150.txt':
            res = 'red mid'
            xy = 20
            lamb = 150
            subp = 247
            outdat = 'outdataredm.txt'
        elif i is 'Red_Low_Spot_Data_4040_300.txt':
            res = 'red high'
            xy = 40
            lamb = 300
            subp = 248
            outdat = 'outdataredh.txt'
        elif i is 'Blue_Low_Spot_Data_4040_300.txt':
            res = 'blue high'
            xy = 40
            lamb = 300
            subp = 244
            outdat = 'outdatablueh.txt'
        
        #data & plot
        #8.5.19
        #catch, mess with order of subplots and share axes
        ax = plt.subplot(subp, sharex = ax)
        plt.title('Interpolation Test:' + res)
        outlam = np.arange(3100, 10000, lstep).astype(float)/10000
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.xlim(-155, -140)
        plt.ylim(-8500, 8500)
        if os.path.isfile(outdat) is True:
            print('Reading...')
            outc = np.genfromtxt(outdat)
            for i in range (100):
                indx = outc[:,2] == i
                plt.scatter(outc[indx,0],outc[indx,1], c = cm.rainbow(outlam), s = 2)
        else:
            print('Generating...')
            fpinterp = WFOSFocalPlane2Detector(i,xy,xy,lamb) 
            nlam = outlam.size
            savec = np.zeros((nlam*nfield,3), dtype = float)
            
            for i,(x,y) in enumerate(zip(random_x, random_y)):
                outxf = np.full(outlam.size, x, dtype=float)
                outyf = np.full(outlam.size, y, dtype=float)
                outc = fpinterp.interp(outxf, outyf, outlam)
                savec[i*nlam:(i+1)*nlam,:2] = outc
                savec[i*nlam:(i+1)*nlam,2] = i
                
                plt.scatter(outc[:,0],outc[:,1], c = cm.rainbow(outlam), s = 2)
            np.savetxt(outdat, savec,  fmt = ["%7.4f", "%7.4f","%3d"])
            
    plt.subplot(241)
    plt.scatter(random_x, random_y, s = 20, c ='k')
    plt.title('Random xf & yf Used')
    plt.xlabel('X (arcmin)')
    plt.ylabel('Y (arcmin)')
    plt.tight_layout()

def tracediff():
    """
    7.23.19
    function that reads text files for interpolated points and plots (and saves difference
    between the different resolutions. need to change filenames if you want to color arm 
    (change blue/red)
    """
    plt.close('all')
    rc('font', size=16)
#result says that function of lambda is ok but as a function of xy it sucks  
    outcbm = np.genfromtxt('outdataredm.txt')
    outcbl = np.genfromtxt('outdataredl.txt')
    outcbh = np.genfromtxt('outdataredh.txt')
    
    xbl = outcbl[:,0]
    ybl = outcbl[:,1]
    
    xbm = outcbm[:,0]
    ybm = outcbm[:,1]
    
    xbh = outcbh[:,0]
    ybh = outcbh[:,1]
    
    yl = ybl.reshape(20,-1)
    ym = ybm.reshape(20,-1)
    nl = outcbl[:,2].reshape(20,-1)
    j = (np.argmax(np.nanmean(yl-ym, axis = 1)))
    
    plt.figure(figsize=(20,13))
    
    plt.subplot(141)
    plt.scatter(xbh-xbm,ybh, s = 2, c = 'r')
    plt.title('X difference Red (high & mid)')
    plt.xlabel('X difference (pixels)')
    plt.ylabel('Y high res (pixels)')
    
    plt.subplot(142)
    plt.scatter(ybh-ybm,ybh, s = 2, c = 'r')
    plt.title('Y difference Red (high & mid)')
    plt.xlabel('Y difference (pixels)')
    plt.ylabel('Y high res (pixels)')
    
    plt.subplot(143)
    plt.scatter(xbh-xbl,ybh, s = 2, c = 'r')
    plt.title('X difference Red (low & high)')
    plt.xlabel('X difference (pixels)')
    plt.ylabel('Y high res (pixels)')
    
    plt.subplot(144)
    plt.scatter(ybh-ybl,ybh, s = 2, c = 'r')
    plt.title('Y difference Red (low & high)')
    plt.xlabel('Y difference (pixels)')
    plt.ylabel('Y high res (pixels)')
    
    plt.tight_layout()
    
    #plt.savefig('tracediff_red.png')
    #plt.savefig('tracediff_red.pdf')

def interpdiff():
    """
    7.23.19
    function that at 3 "random" preselected field points (xf,yf) that must exist within the high
    res grid but not the mid or low res, generates the interpolator at either low or mid res 
    (specify in fpinterp) and plots (and saves)comparison between the interpolated points and 
    actual coordinates taken from the high res.
    
    need to change filenames if you want to color arm (change blue/red) or switch low/mid res
    """
    plt.close('all')
    rc('font', size=16)
    
    #3 random points
    xf = np.array([2.05, -1.83, 1.83])
    yf = np.array([0.81, -1.42, -0.35])
 
    #get the accurate high points, change file name depending on red/blue
    modelfile = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Blue_Low_Spot_Data_4040_300.txt')
    db = np.genfromtxt(modelfile)
    
    save = np.zeros((0,5), dtype = float)

    for i in range(3):
        indx = (db[:,0]==xf[i]) & (db[:,1]==yf[i])
        save = np.append(save, db[indx,:5], axis = 0)
    
    #specify interp setup depending on low/mid res
    fpinterp = WFOSFocalPlane2Detector('Blue_Low_Spot_Data_2020_150.txt',20,20,150)
    
    outxf = save[:,0]
    outyf = save[:,1]
    outlam = save[:,2]
    outxc = save[:,3]/0.015
    outyc = save[:,4]/0.015
    
    outc = fpinterp.interp(outxf, outyf, outlam)
    fig , ax1 = plt.subplots(1,1, figsize=(10,10))
    xlim_pix = np.array([-5,3])
    ylim_pix = np.array([-3,10])

    plt.xlabel('Difference in X (pixels)')
    plt.ylabel('Difference in Y (pixels)')
    ax1.scatter(outc[:,0]-outxc, outc[:,1]-outyc, s = 20, alpha = 0.5, c='b')
    
    ax2 = ax1.twinx()
    ax2.set_ylim(ylim_pix/16000)
    ax2.set_ylabel(r'$\Delta Y$')
    
    ax3 = ax1.twiny()
    ax3.set_xlim(xlim_pix/10000)
    ax3.set_xlabel(r'$\Delta X$')
    plt.title('3 Point Interpolation VS Truth (Blue Mid)', y=1.1)
    plt.tight_layout()
    #plt.savefig('interpdiff_rm.png')
    #plt.savefig('interpdiff_rm.pdf')
    
def interpfig():
    plt.close('all')
    
    #load & parse
    dbh = np.genfromtxt(os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Blue_Low_Spot_Data_4040_300.txt'))
    dbm = np.genfromtxt(os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Blue_Low_Spot_Data_2020_150.txt'))
    dbl = np.genfromtxt(os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Blue_Low_Res_Spot_Data_9x9F_90W.txt'))
    
    xfh = dbh[:,0]
    yfh = dbh[:,1]

    xfm = dbm[:,0]
    yfm = dbm[:,1]
    

    xfu = np.setdiff1d(xfh,xfm)
    yfu = np.setdiff1d(yfh,yfm)

    #get the accurate high points, change file name depending on red/blue
    save = np.zeros((0,5), dtype = float)

    for i in range(38):
        indx = (dbh[:,0]==xfu[i]) & (dbh[:,1]==yfu[i])
        save = np.append(save, dbh[indx,:5], axis = 0)
    
    #specify interp setup depending on low/mid res
    fpinterpm = WFOSFocalPlane2Detector('Blue_Low_Spot_Data_2020_150.txt',20,20,150)
    fpinterpl = WFOSFocalPlane2Detector('Blue_Low_Res_Spot_Data_9x9F_90W.txt',9,9,93)
    
    outxf = save[:,0]
    outyf = save[:,1]
    outlam = save[:,2]
    outxc = save[:,3]/0.015
    outyc = save[:,4]/0.015
    
    outcm = fpinterpm.interp(outxf, outyf, outlam)
    outcl = fpinterpl.interp(outxf, outyf, outlam)
    
    #rc('font', size=13)
    
    fig, ax1 = plt.subplots(1,1, figsize = (12,12))
    
    plt.xlabel('Difference in X (pixels)')
    plt.ylabel('Difference in Y (pixels)')
    ax1.scatter(outxc - outcm[:,0], outyc - outcm[:,1], s = 20, alpha = 0.5, lw = 0, marker = 'o', c='purple', label = 'Meduim Res')
    ax1.scatter(outxc - outcl[:,0], outyc - outcl[:,1], s = 20, alpha = 0.5, lw = 0, marker = 'o', c='orange', label = 'Low Res')
    ax1.legend(['Meduim Res' , 'Low Res'])
    xlim_pix = np.array([-10,10])
    ylim_pix = np.array([-10,10])
     
    ax2 = ax1.twinx()
    ax2.set_ylim(ylim_pix/16000)
    ax2.set_ylabel(r'$\Delta Y$')
    
    ax3 = ax1.twiny()
    ax3.set_xlim(xlim_pix/10000)
    ax3.set_xlabel(r'$\Delta X$')
    plt.title('Interpolation VS Truth (Blue Cam)', y=1.1)
    
    plt.tight_layout()
    
def testraysthru():
    
    plt.close('all')
    rc('font', size=16)
    
    #3 random points
    xf = np.array([2.05, -1.83, 1.83])
    yf = np.array([0.81, -1.42, -0.35])
    
    #get the accurate high points, change file name depending on red/blue
    modelfile = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Red_Low_Spot_Data_4040_300.txt')
    db = np.genfromtxt(modelfile)
    
    save = np.zeros((0,6), dtype = float)

    for i in range(3):
        indx = (db[:,0]==xf[i]) & (db[:,1]==yf[i])
        save = np.append(save, db[indx,:6], axis = 0)
    
    #specify interp setup depending on low/mid res
    fpinterp = WFOSFocalPlane2Detector('Red_Low_Res_Spot_Data_9x9F_90W.txt',9,9,93)
    fpinterpm = WFOSFocalPlane2Detector('Red_Low_Spot_Data_2020_150.txt',20,20,150)
    
    outxf = save[:,0]
    outyf = save[:,1]
    outlam = save[:,2]
    outxc = save[:,3]/0.015
    outyc = save[:,4]/0.015
    outraysthru = save[:,5]
    
    outc = fpinterp.interpraysthru(outxf, outyf, outlam)
    outcm = fpinterpm.interpraysthru(outxf, outyf, outlam)
    
    print(outlam.shape)

    plt.figure(figsize=(20,7.5))
    plt.title('Rays Thru Test')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('% Rays Through')
    plt.plot(outlam[:300], outraysthru[:300], c='k', label = '<300')
    plt.plot(outlam[:300], outc[:300,2], c='r', label = 'low')
    plt.plot(outlam[:300], outcm[:300,2], c='g', label = 'mid')

    plt.plot(outlam[300:600], outraysthru[300:600], c ='k', linestyle = '--',label = '300-600')
    plt.plot(outlam[300:600], outc[300:600,2], c='r', label = 'low')
    plt.plot(outlam[300:600], outcm[300:600,2], c='g', label = 'mid')

    plt.plot(outlam[600:], outraysthru[600:], c = 'k', linestyle = ':', label = '>600')
    plt.plot(outlam[600:], outc[600:,2], c='r', label = 'low')
    plt.plot(outlam[600:], outcm[600:,2], c='g', label = 'mid')
    #add legend, save plot, add to google doc
    plt.legend()


def testinvert():
    """
    7.31.19
    test setup for invinterp
    
    take xf yf lam (at some point), interp to get xc and yc, then use xc yc yf to invinterp get xf and lam, check if xf here and firs xf are the same
    """
    plt.close('all')

    #3 random points
    xf = np.array([2.05, -1.83, 1.83])
    yf = np.array([0.81, -1.42, -0.35]) 
        
    #get the accurate high points, change file name depending on red/blue
    modelfile = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Red_Low_Spot_Data_4040_300.txt')
    db = np.genfromtxt(modelfile)
    
    save = np.zeros((0,6), dtype = float)

    for i in range(3):
        indx = (db[:,0]==xf[i]) & (db[:,1]==yf[i])
        save = np.append(save, db[indx,:6], axis = 0)
    
    #specify interp setup depending on low/mid res
    fpinterp = WFOSFocalPlane2Detector('Red_Low_Res_Spot_Data_9x9F_90W.txt',9,9,93)
    #fpinterpm = WFOSFocalPlane2Detector('Red_Low_Spot_Data_2020_150.txt',20,20,150)
    
    outxf = save[:,0]
    outyf = save[:,1]
    outlam = save[:,2]
    outxc = save[:,3]/0.015
    outyc = save[:,4]/0.015
    outraysthru = save[:,5]
    
    #uhhhh
    outc1 = fpinterp.interp(outxf, outyf, outlam)
    
    outc2 = fpinterp.invinterp(outc1[:,0], outc1[:,1], outyf)
      
    plt.scatter(outc2[:,0] - outxf, outc2[:,1] - outlam, s = 2)
    plt.title('Accuracy of Interp > Inv Interp > ?')
    plt.xlabel('XF (pixels)')
    plt.ylabel('Wavelength (microns)')
    
    """
    #data & plot
    plt.xlabel('XF (pixels)')
    plt.ylabel('Wavelength (microns)')

    outc = fpinterp.invinterp(outxc, outyc, outyf)
    plt.scatter(outc[:,0], outc[:,1], c = 'r', s = 2)
    plt.scatter(outxf, outlam, c = 'k', s = 2)
    """
def example():
    plt.close('all')
    plt.figure()
    plt.title('1D Interpolation Example')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    ax = np.arange(0,20,0.1)

    x = np.linspace(0, 20, num=25)
    y = np.sin(x)
    f = interpolate.interp1d(x, y)

    xnew = np.linspace(0, 20, num=41, endpoint=True)

    plt.plot(ax, np.sin(ax), '-',c='k')
    plt.plot(x, y, 'o', c='purple')
    plt.plot(xnew, f(xnew), '--',c='purple')
    plt.show()

#need to pick which function you want to run here
#note: older tests are located in "old functions.py"
if __name__ == '__main__':
    example()

  