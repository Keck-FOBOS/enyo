# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:02:44 2019

@author: sueel
"""

import enyo
import numpy as np
import os
import math
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
from scipy import interpolate

plt.close("all")

def interp (step, lstep, start, ls):
    """
    what: function that linearly interpolates with data from a file
    where: step is desired steps IN
           start is desired location in array to start at (to be interpolated)
           ls is the wavelength integer address number (out of total unique wavelengths) 
           to interpolate at
    returns: interpolated array of points y, modified array of actual points out_c, ls
    to change: make get data more robust so that file name is an input (7.1.19)
    """
    #get data
    modelfile_b = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Blue_Low_Res_Spot_Data_9x9F_90W.txt')
    #modelfile_r = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Red_Low_Res_Spot_Data_9x9F_90W.txt')
    db_b = np.genfromtxt(modelfile_b)
    #db_r = np.genfromtxt(modelfile_r)
    
    #set up boolean grid
    nrows = 9*9*93
    row = np.arange(nrows).reshape(9,9,93)
    interprow = np.unique(np.append([row[0,0,0],row[8,0,0],row[0,8,0],row[0,0,92],row[0,8,92],row[8,8,0],row[8,8,92],row[8,0,92]],row[step::step,step::step,lstep::lstep])).ravel()
    use_row_for_grid = np.zeros(nrows, dtype='bool')
    use_row_for_grid[interprow] = True
    
    #set up arrays to send IN
    inxf_b = db_b[use_row_for_grid,0]/0.015
    inyf_b = db_b[use_row_for_grid,1]/0.015
    inlam_b = db_b[use_row_for_grid,2]
    inxc_b = db_b[use_row_for_grid,3]/0.015
    inyc_b = db_b[use_row_for_grid,4]/0.015

    #set up arrays for OUT
    use_row_for_test = np.invert(use_row_for_grid)
    outxf_b = db_b[use_row_for_test,0]/0.015
    outyf_b = db_b[use_row_for_test,1]/0.015
    outlam_b = db_b[use_row_for_test,2]
    outxc_b = db_b[use_row_for_test,3]/0.015
    outyc_b = db_b[use_row_for_test,4]/0.015
    
    lu = np.unique(outlam_b)
    lam_for_plot = outlam_b == lu[ls]

    #f, c, & xi for NDInterp and transpose (.T)
    f = np.array([inxf_b.ravel(), inyf_b.ravel(),inlam_b.ravel()]).T
    
    c = np.array([ inxc_b.ravel(), inyc_b.ravel()]).T
    out_c = np.array([outxc_b.ravel(), outyc_b.ravel()]).T
    
    xi = np.array([outxf_b.ravel(), outyf_b.ravel(), outlam_b.ravel()]).T

    #interp
    interpND = interpolate.LinearNDInterpolator(f,c)
    y = interpND(xi)
    
    return out_c, y, lu, ls, lam_for_plot, step

def interp1 (step, lstep):
    """
    what: function that linearly interpolates with data from a file
    where: step is desired step for x & y
           ls step is desired step for lambda
    returns: interpolated array of points y, modified array of actual points out_c
    to change: make get data more robust so that file name is an input (7.1.19)?
    """
    #get data
    modelfile_b = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Blue_Low_Res_Spot_Data_9x9F_90W.txt')
    #modelfile_r = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Red_Low_Res_Spot_Data_9x9F_90W.txt')
    db_b = np.genfromtxt(modelfile_b)
    #db_r = np.genfromtxt(modelfile_r)
    
    #set up boolean grid
    nrows = 9*9*93
    row = np.arange(nrows).reshape(9,9,93)
    interprow = np.unique(np.append([row[0,0,0],row[8,0,0],row[0,8,0],row[0,0,92],row[0,8,92],row[8,8,0],row[8,8,92],row[8,0,92]],row[step::step,step::step,lstep::lstep].ravel()))
    waverow = list(row[:,:,:].ravel())
    #waverow = np.array(list((set(waverow.ravel().tolist())-set(interprow.ravel().tolist()))))
    for i in waverow:
        if i in interprow:
            waverow.remove(i)
    use_row_for_grid = np.zeros(nrows, dtype='bool')
    use_row_for_grid[interprow] = True
    use_row_for_grid &= (db_b[:,-1]>0)
    
    #set up arrays to send IN
    inxf_b = db_b[use_row_for_grid,0]/0.015
    inyf_b = db_b[use_row_for_grid,1]/0.015
    inlam_b = db_b[use_row_for_grid,2]
    inxc_b = db_b[use_row_for_grid,3]/0.015
    inyc_b = db_b[use_row_for_grid,4]/0.015

    #set up arrays for OUT
    use_row_for_test = np.invert(use_row_for_grid) & (db_b[:,-1]>0)
    outxf_b = db_b[waverow,0]/0.015
    outyf_b = db_b[waverow,1]/0.015
    outlam_b = db_b[waverow,2]
    outxc_b = db_b[waverow,3]/0.015
    outyc_b = db_b[waverow,4]/0.015
    
    #f, c, & xi for NDInterp and transpose (.T)
    f = np.array([inxf_b.ravel(), inyf_b.ravel(),inlam_b.ravel()]).T
    
    c = np.array([ inxc_b.ravel(), inyc_b.ravel()]).T
    out_c = np.array([outxc_b.ravel(), outyc_b.ravel()]).T
    
    xi = np.array([outxf_b.ravel(), outyf_b.ravel(), outlam_b.ravel()]).T

    #interp
    interpND = interpolate.LinearNDInterpolator(f,c)
    y = interpND(xi)
    
    return out_c, y, outlam_b, step, outxf_b, outyf_b

def plot_interp(out_c, y, ls, step):
    """
    what: function to plot data points alongside interpolated points as well as difference
          between point values
    returns: distance array
    """
    plt.figure()
    plt.title('Lambda/Wavelength = ' )#+ str(lu[ls]) + ' Microns, Grid Resolution = ' +str(step))
    plt.xlabel('X Coord (pixels)')
    plt.ylabel('Y Coord (pixels)')
    plt.scatter(out_c[:,0],out_c[:,1], s=3, c='BLue',alpha=0.5)
    plt.scatter(y[:,0],y[:,1], s=3, c='Red',alpha=0.5)
    plt.grid()
    plt.legend(['Interpolated Points','Data Points'])
    
    #distance between truth and interp points to determine overall interp accuracy
    dist = []
    plt.figure()
    for i in range (out_c[:,0].size):
        distx = math.pow((out_c[i,0]-y[i,0]),2)
        disty = math.pow((out_c[i,1]-y[i,1]),2)
        dist = np.append(dist, (np.sqrt(distx+disty)))
    plt.ylim(0,np.max(dist)+5)
    plt.title('Interpolation Accuracy (Lambda = ' )#+ str(lu[ls]) + ' Microns)')
    plt.xlabel('Point Number')
    plt.ylabel('Difference (pixels)')
    plt.scatter(np.arange((out_c[:,0]).size), dist)
    dist = np.nanmean(dist)
    return dist

def distance(out_c, y):
    #distance between truth and interp points to determine interp accuracy
    dist = []
    for i in range (out_c[:,0].size):
        distx = math.pow((out_c[i,0]-y[i,0]),2)
        disty = math.pow((out_c[i,1]-y[i,1]),2)
        dist = np.append(dist, (np.sqrt(distx+disty)))
    #dist = np.nanmedian(dist)
    return dist

def sigma(values):
    dx = sigma_clip(values)
    meand = np.ma.mean(dx)
    stdd = np.ma.std(dx)
    sumd = np.sum(dx.mask)
    return meand, stdd, sumd

#main funct
if __name__ == '__main__':
    """
    out_c, y,ls,step = interp1(2,2,25)
    d = distance(out_c,y)
    meand, stdd, sumd = sigma(d)
    plt.scatter(out_c[:,0], out_c[:,1],s=2)
    plt.scatter(y[:,0],y[:,1],s=2)
    
    meanplt = []
    stdplt = []
    sumplt = []
    
    out_c, y, lam, outxf_b, outyf_b = interp1(2,2)
    nd = out_c.shape[0]
    nrf = 9-2
    
    Df = np.zeros((nd,nrf),dtype='float')
    La = np.zeros((nd,nrf),dtype='float')
    delt = np.zeros((nd,nrf),dtype='float')
    rf = np.zeros((nd,nrf),dtype='float')
    ma = np.zeros((nd,nrf),dtype='bool')
    
    Df[:nd,0] = np.sqrt(np.square(outxf_b) + np.square(outyf_b))
    La[:nd,0] = lam
    rf[:,0] = 0
    delt[:nd,0] = distance(out_c, y)
    ma[nd:,0] = True
    #
    """
    nrf = 9-2
    redf = np.arange(2,9)
    Df = np.zeros((0,nrf),dtype='float')
    La = np.zeros((0,nrf),dtype='float')
    delt = np.zeros((0,nrf),dtype='float')
    rf = np.zeros((0,nrf),dtype='float')
    ma = np.zeros((0,nrf),dtype='bool')
    
    for i in redf:
        out_c, y, lam, step, outxf_b, outyf_b = interp1(i,i)

        nd = out_c.shape[0]
        if Df.shape[0] < nd:
            Df = np.vstack((Df,np.zeros((nd-Df.shape[0],nrf),dtype='float')))
            La = np.vstack((La,np.zeros((nd-La.shape[0],nrf),dtype='float')))
            delt = np.vstack((delt,np.zeros((nd-delt.shape[0],nrf),dtype='float')))
            rf = np.vstack((rf,np.zeros((nd-rf.shape[0],nrf),dtype='float')))
            ma = np.vstack((ma,np.zeros((nd-ma.shape[0],nrf),dtype='bool')))
            
        Df[:nd,i-2] = np.sqrt(np.square(outxf_b) + np.square(outyf_b))
        La[:nd,i-2] = lam
        rf[:,i-2] = i
        delt[:nd,i-2] = distance(out_c, y)
        ma[nd:,i-2] = True
 
    #print(np.sum(np.invert(np.isfinite(delt))))
    #print(delt.size)
    
    for i in range(nrf):
        df_u = np.unique(Df)
        mdelt = []
        for j,h in enumerate(df_u):
            indx = (Df[:,i] == h) & np.invert(ma[:,i])
            mdelt += [np.nanmean(delt[indx,i])]
        plt.plot(df_u, mdelt, label = ('Reduction Factor = ' +str(i+2)))
        plt.legend()
    plt.title('Mean Delta Correlation- RED ARM')
    plt.xlabel('Df (pixels)')
    plt.ylabel('Mean Delta (pixels)')
    #plt.savefig('dist vs meandelt RED.jpg')