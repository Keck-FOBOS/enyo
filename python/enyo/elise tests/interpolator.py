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
import matplotlib.pyplot as plt
from matplotlib import colors, cm, figure
from scipy import interpolate

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
        f = np.array([inxf.ravel(), inyf.ravel(), inlam.ravel()]).T
        
        c = np.array([ inxc.ravel(), inyc.ravel()]).T
        
        #define interp/data arrange outside class??
        self.interpolator = interpolate.LinearNDInterpolator(f,c)
        self.gridsize = f.shape[0]

    #make it just call self.interpolator, pass it xf yf lam to interp @, returns interpolated 
    def interp(self, outxf, outyf, outlam):
    
        xi = np.array([outxf.ravel(), outyf.ravel(), outlam.ravel()]).T

        return self.interpolator(xi)
     
class WFOSBlueFocalPlane2Detector(FocalPlane2Detector):
    def __init__(self,filename,step=None):
        whatfile = 'Blue_Low_Res_Spot_Data_9x9F_90W.txt'
        modelfile = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', whatfile)
        self.db = np.genfromtxt(modelfile)
        
        nx = 9
        ny = 9
        nl = 93
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
        
        inxf, inyf, inlam, inxc, inyc = grabdata(self.db,self.use_row_for_grid)
        
        super(WFOSBlueFocalPlane2Detector,self).__init__(inxf, inyf, inlam, inxc, inyc)
        self.gridfrac = self.gridsize/self.db.shape[0]
        #print(self.gridfrac)
        
class WFOSRedFocalPlane2Detector(FocalPlane2Detector):
    def __init__(self,step):
        whatfile = 'Red_Low_Res_Spot_Data_9x9F_90W.txt'
        modelfile = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', whatfile)
        self.db = np.genfromtxt(modelfile)
        
        nx = 9
        ny = 9
        nl = 93
        nrows = nx*ny*nl
        row = np.arange(nrows).reshape(nx,ny,nl)
    
        # Get the rows to use in the interpolator
        thinned_rows = grid_thin_indx(row, step)

        self.use_row_for_grid = np.zeros(row.size, dtype=bool)
        self.use_row_for_grid[thinned_rows] = True
        
        self.use_row_for_grid &= (self.db[:,-1] > 0)
        
        inxf, inyf, inlam, inxc, inyc = grabdata(self.db,self.use_row_for_grid)
        
        super(WFOSRedFocalPlane2Detector,self).__init__(inxf, inyf, inlam, inxc, inyc)
        self.gridfrac = self.gridsize/self.db.shape[0]

#"main" function     
def main1():
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
        #print(fpinterp.gridfrac)
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
        #print(redf[i])
        #print(mdelt)
        plt.plot(df_u.compressed(), np.array(mdelt)[np.invert(df_u.mask)], color = (colors[i].tolist()), label = ('Reduction Factor = ' +str(redf[i])))
        #print(df_u)
        #plt.plot(df_u, mdelt, color = tuple(colors[i].tolist()), label = ('Reduction Factor = ' +str(redf[i])))
        plt.legend()
    plt.title('Mean Delta Correlation- RED ARM')
    plt.xlabel('lam')
    plt.ylabel('Mean Delta (pixels)')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    #plt.savefig('NEW df vs meandelt RED.jpg')
    """
    plt.figure()
    for i in index:
        #Df
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
    """
    
def main2():
    """
    1. define 6 interps outside loop
    2.generate points
    3. for each point, construct xf,yf for each lambda
    4. call all 6 to get the plots
    
    set xf yf lam to 0 make 1 plot
    
    pick 100 random points within the field (xf,yf), going to send to devika to ask her for the actual values at these points
    for wavelength, generate vector that goes from 3100 - 10000 ang in steps of 1 angstrom
    
    
    
    with the points, save them
    
    build the interpolator using the 3 different versions (low, mid, high res), all points
    
    use interp to predict where the 100 points are
    
    look @ difference between predictions for all 3 res (theres no actual yet)
    
    """  
#call main function so it doesn't try to run without variables
if __name__ == '__main__':
    main()
    
        
  