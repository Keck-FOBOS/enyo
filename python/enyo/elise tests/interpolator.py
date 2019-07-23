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
     return inxf,inyf,inlam,inxc,inyc
 
    
#FP2D class & Red/Blue derived classes
class FocalPlane2Detector:
    #only work with data directly
    def __init__(self, inxf, inyf, inlam, inxc, inyc):
        """
        init shoud:
            generate elements from file
            set up interpolator
        """    
        #f, c, & xi for NDInterp and transpose (.T)
        self.f = np.array([inxf.ravel(), inyf.ravel(), inlam.ravel()]).T
        
        self.c = np.array([ inxc.ravel(), inyc.ravel()]).T
        
        self.interpolator = None
        
        self.gridsize = self.f.shape[0]

    #make it just call self.interpolator, pass it xf yf lam to interp @, returns interpolated 
    def interp(self, outxf, outyf, outlam):
        if self.interpolator is None:    
            self.interpolator = interpolate.LinearNDInterpolator(self.f,self.c)
            
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
        
        inxf, inyf, inlam, self.inxc, self.inyc = grabdata(self.db,self.use_row_for_grid)
        
        super(WFOSFocalPlane2Detector,self).__init__(inxf, inyf, inlam, self.inxc, self.inyc)
        self.gridfrac = self.gridsize/self.db.shape[0]

  
def main():
    """
    7.23.19
    old function that plots meandelta as a function of lambda or distance
    need to change fpinterp to work
    """
    plt.close('all')
    
    nx = 9
    ny = 9
    nl = 93
    nrows = nx*ny*nl
    row = np.arange(nrows).reshape(nx,ny,nl)
    redf = iteration_steps(row)
    nrf = redf.size

    Df = np.zeros((0,nrf),dtype='float')
    La = np.zeros((0,nrf),dtype='float')
    delt = np.zeros((0,nrf),dtype='float')
    rf = np.zeros((0,nrf),dtype='float')
    ma = np.zeros((0,nrf),dtype='bool')
    
    for j,i in enumerate(redf):
        fpinterp = WFOSRedFocalPlane2Detector(i)
        outxf, outyf, outlam, outxc, outyc = \
                grabdata(fpinterp.db,np.invert(fpinterp.use_row_for_grid))
        out_c = np.array([outxc.ravel(), outyc.ravel()]).T
        y = fpinterp.interp(outxf,outyf,outlam)
        nd = out_c.shape[0]

        if Df.shape[0] < nd:
            Df = np.vstack((Df,np.zeros((nd-Df.shape[0],nrf),dtype='float')))
            La = np.vstack((La,np.zeros((nd-La.shape[0],nrf),dtype='float')))
            delt = np.vstack((delt,np.zeros((nd-delt.shape[0],nrf),dtype='float')))
            rf = np.vstack((rf,np.zeros((nd-rf.shape[0],nrf),dtype='float')))
            ma = np.vstack((ma,np.zeros((nd-ma.shape[0],nrf),dtype='bool')))
        Df[:nd,j] = np.sqrt(np.square(outxf) + np.square(outyf))
        La[:nd,j] = outlam
        rf[:,j] = fpinterp.gridfrac
        delt[:nd,j] = distance(out_c, y)
        ma[nd:,j] = True   
 
    #only keep unique rf/gridfrac values so there arent extra iterations
    unique, index = np.unique(rf[0,:],return_index = True)
    colors = np.empty((rf.shape[1],4), dtype = object)
    colors[index,:] = cm.get_cmap('seismic')(np.linspace(0,1,index.size))
    
    for i in index:
        #La
        df_u = np.ma.MaskedArray(np.unique(La))
        mdelt = []
        for j,h in enumerate(df_u):
            indx = (La[:,i] == h) & np.invert(ma[:,i]) & np.isfinite(delt[:,i])
            if not np.any(indx):
                df_u[j] = np.ma.masked
                mdelt += [0]
                continue
            mdelt += [np.nanmean(delt[indx,i])]
        plt.plot(df_u.compressed(), np.array(mdelt)[np.invert(df_u.mask)], color = (colors[i].tolist()), label = ('Reduction Factor = ' +str(redf[i])))
        #plt.plot(df_u, mdelt, color = tuple(colors[i].tolist()), label = ('Reduction Factor = ' +str(redf[i])))
        plt.legend()
    plt.title('Mean Delta Correlation- RED ARM')
    plt.xlabel('lam')
    plt.ylabel('Mean Delta (pixels)')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    #plt.savefig('NEW df vs meandelt RED.jpg')
    plt.figure()
    
    for i in index:
        df_u = np.ma.MaskedArray(np.unique(Df))
        mdelt = []
        for j,h in enumerate(df_u):
            indx = (Df[:,i] == h) & np.invert(ma[:,i]) & np.isfinite(delt[:,i])
            if not np.any(indx):
                df_u[j] = np.ma.masked
                mdelt += [0]
                continue
            mdelt += [np.nanmean(delt[indx,i])]
        plt.plot(df_u, mdelt, color = colors[i], label = ('Reduction Factor = ' +str(redf[i])))
        plt.legend()
    plt.title('Mean Delta Correlation- RED ARM')
    plt.xlabel('lam')
    plt.ylabel('Mean Delta (pixels)')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    #plt.savefig('NEW df vs meandelt RED.jpg')
    
    
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
        plt.subplot(subp)
        plt.title('Interpolation Test:' + res)
        outlam = np.arange(3100, 10000, lstep).astype(float)/10000
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        
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
    plt.scatter(random_x, random_y, s = 2)
    plt.title('Random xf & yf Used')
    plt.xlabel('X (arcmin)')
    plt.ylabel('Y (arcmin)')
    plt.tight_layout()
    
    #plt.savefig('trace.png')
    #plt.savefig('trace.pdf')


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
    modelfile = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Red_Low_Spot_Data_4040_300.txt')
    db = np.genfromtxt(modelfile)
    
    save = np.zeros((0,5), dtype = float)

    for i in range(3):
        indx = (db[:,0]==xf[i]) & (db[:,1]==yf[i])
        save = np.append(save, db[indx,:5], axis = 0)
    
    #specify interp setup depending on low/mid res
    fpinterp = WFOSFocalPlane2Detector('Red_Low_Spot_Data_2020_150.txt',20,20,150)
    
    outxf = save[:,0]
    outyf = save[:,1]
    outlam = save[:,2]
    outxc = save[:,3]/0.015
    outyc = save[:,4]/0.015
    
    outc = fpinterp.interp(outxf, outyf, outlam)
    plt.figure(figsize=(20,7.5))
    plt.title('Difference between interpolated & actual (red m)')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.scatter(outc[:,0]-outxc, outc[:,1]-outyc, s = 2, c='r')
    
    #plt.savefig('interpdiff_rm.png')
    #plt.savefig('interpdiff_rm.pdf')
    
#need to pick which function you want to run here
if __name__ == '__main__':
    tracefig()


        
  