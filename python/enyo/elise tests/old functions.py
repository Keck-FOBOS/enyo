# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:54:48 2019

@author: sueel
"""

#practice 6.26.19: interpolate where yf and xf both = 0
    #indices for zeroed xf,yf (to plot)
interindx = (xf_b == 0) & (yf_b == 0) & (yc_b > 0) & (lam_b > 0)
inyc_b = yc_b[interindx]
inlam_b = lam_b[interindx]

#scatter plot of lambda and yc where xf and yf are 0, start at 0 increment 10
plt.xlabel('Wavelength/Lambda (micron)')
plt.ylabel('Y Coord (mm)')
plt.scatter(inlam_b[::10],inyc_b[::10], s=3, c='Blue',)
   
#interpolation over all lam_b and yc_b values
f = interpolate.interp1d(inlam_b[::10], inyc_b[::10], bounds_error=False)

#to compare
newlam_b = inlam_b[5::10]
    #yc interpolated with newlam, offset for display purpose
newyc_b = f(newlam_b)
    #yc interpolated with inlam, not offset for difference calc
comyc_b = f(inlam_b)
plt.scatter(newlam_b, newyc_b, s=3, c='Red')
plt.scatter(newlam_b,inyc_b[5::10], s=3, c='Purple')

#alternative interp 6.27.19
    interpRy = interpolate.Rbf(in_yf.ravel(),in_xf.ravel(),in_lam.ravel(),in_yc.ravel())
    interpRx = interpolate.Rbf(in_yf.ravel(),in_xf.ravel(),in_lam.ravel(),in_xc.ravel())
    y = interpRy(out_yf.ravel(), out_xf.ravel(), out_lam.ravel())
    x = interpRx(out_yf.ravel(), out_xf.ravel(), out_lam.ravel())

#obsolete plot function 6.27.19
def plot_lininterp(xf_b, yf_b, lam_b, xc_b, yc_b, per_b, yu_b, xu_b):    
    meandiff = np.zeros(81).reshape(9,9)
    for i in range (yu_b.size):
        yindx = yf_b == yu_b[i]
        for j in range (xu_b.size):
            indx = (xf_b == xu_b[j]) & yindx & (per_b != 0)    
            
            inyc_b = yc_b[indx]
            inlam_b = lam_b[indx]
        
            f = interpolate.interp1d(inlam_b[::10], inyc_b[::10], bounds_error=False)
            
            newlam_b = inlam_b[5::10]
            newyc_b = f(newlam_b)
            
            plt.ylim(0,0.7)
            plt.xlim(-125,125)
            plt.title('Interpolation Practice Field No.' + str(i+1))
            plt.xlabel('Wavelength/Lambda (micron)')
            plt.ylabel('Y Coord (mm)')
            plt.scatter(inyc_b,inlam_b, s=2, c='Blue')
            plt.scatter(newyc_b, newlam_b, s=3, c='Red')
               
            plt.figure()
            plt.title('Comparison No.' + str(i+1))
            plt.xlabel('Y Coord (mm)')
            plt.ylabel('Difference between interpolated and actual (mm)')

            meandiff[i,j]=np.mean(inyc_b[5::10]-newyc_b)
    plt.scatter(np.arange(meandiff.size),meandiff)
    #generation of xc yc plots based off of xf yf inputs
    plt.scatter(xc_b[indx], yc_b[indx], c = lam_b[indx], s = 2, cmap = 'winter')
    plt.xlim(-150,150)
    plt.ylim(-150,150)
    plt.xlabel('X Coord (mm)')
    plt.ylabel('Y Coord (mm)')
    plt.title('Spot Graphing Test No.'+ str(i+1) + 'Blue')
    #plt.savefig('test' + str(i+1) + 'b.png')
    plt.figure()
    plt.show()
    return

 #get data function that got built into interp 6.28.19
def get_data():
    """
    what: function to get data from file, separates important values
    returns: separated arrays of values
    """
    #locate model files and extract data from files
    modelfile_b = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Blue_Low_Res_Spot_Data_9x9F_90W.txt')
    modelfile_r = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', 'Red_Low_Res_Spot_Data_9x9F_90W.txt')
    db_b = np.genfromtxt(modelfile_b)
    db_r = np.genfromtxt(modelfile_r)
    
    #parse: declare variables from files, Kick out 0's and Pick out unique points (blue)
    #reshape and convert to pixels (could also convert wavelength to angstroms if desired)
    xf_b = db_b[:,0].reshape(9,9,93)/0.015
    yf_b = db_b[:,1].reshape(9,9,93)/0.015
    lam_b = db_b[:,2]
    xc_b = db_b[:,3].reshape(9,9,93)/0.015
    yc_b = db_b[:,4].reshape(9,9,93)/0.015
    per_b = db_b[:,5]
    xu_b = np.unique(xf_b)
    yu_b = np.unique(yf_b)

    return xf_b, yf_b, lam_b, xc_b, yc_b, per_b, yu_b, xu_b, db_b

#stuff in main from 7.1.19
    """
    #now we can pass any desired xd, yd, ld, xs, ys, ls to interp and plot
    out_c, y, lu, ls, lfp, step = interp(4,10,2,65)
    d = plot_interp(out_c,y,lu,ls,lfp,step)
    
    plt.figure()
    for j in [25,65,90]:
        d = []
        for i in range(2,9):
            out_c, y, lu, ls, lfp, step = interp(i,i,2,65)            
            d = np.append(d, distance(out_c,y))
        plt.plot(d+j)
    plt.title('?')
    plt.ylabel('Median Distances')
    plt.xlabel('Reduction Factor')

    meanplt = []
    stdplt = []
    sumplt = []
    medianplt = []
    plt.figure
    
    for i in range(2,9):
        out_c, y,ls,step = interp1(i,i)
        d = distance(out_c,y)
        
        plt.figure()
        plt.scatter(out_c[:,0], out_c[:,1],s=2)
        plt.scatter(y[:,0],y[:,1],s=2)
        plt.grid()
        plt.xlabel('X Coord (pixels)')
        plt.ylabel('Y Coord (pixels)')
        plt.legend(['Interpolated Points','Data Points'])
        
        meand,stdd,sumd = sigma(d)
        meanplt = np.append(meanplt,meand)
        stdplt = np.append(stdplt,stdd)
        sumplt = np.append(sumplt,sumd)
    plt.figure()
    plt.plot(meanplt)
    plt.plot(stdplt)
    plt.plot(sumplt)
    plt.title('?')
    plt.ylabel('Median Distances Sigma Clipped (pixels)')
    plt.xlabel('Reduction Factor')
    plt.legend(['Mean Values','Standard Deviations','Sum of Points Rejected'])
    """
#7.12.19 old gridrows and OLDoldgridrows
def oldgridrows(db,step,lstep, flip):
    #set up boolean grid 
    
    nrows = 9*9*93
    row = np.arange(nrows).reshape(9,9,93)
    corners = np.array([row[0,0,0],row[8,0,0],row[0,8,0],row[0,0,92],row[0,8,92],row[8,8,0],row[8,8,92],row[8,0,92]])
    use_row_for_grid = np.zeros(nrows, dtype='bool')
    use_row_for_grid[corners] = True
    interprow = row[step::step,step::step,lstep::lstep].ravel()
    print(len(interprow))
    print(step)
    
    internalpts = np.zeros(nrows, dtype='bool')
    internalpts[interprow] = True
    
    if flip:
        internalpts = np.invert(internalpts)
    use_row_for_grid |= internalpts
    use_row_for_grid &= (db[:,-1]>0)
    
    if not np.any(use_row_for_grid):
        raise ValueError()
    return use_row_for_grid

def gridrows(db,step,lstep,flip):
    
    corners = np.concatenate(([row[0,0,-1]], [row[0,0,0]], np.atleast_1d(row[0,0,nl//2::nl//lstep]), np.atleast_1d(row[0,0,nl//2::-(nl//lstep)]),
                        [row[-1,0,-1]], [row[-1,0,0]], np.atleast_1d(row[-1,0,nl//2::nl//lstep]), np.atleast_1d(row[-1,0,nl//2::-(nl//lstep)]),
                        [row[0,-1,-1]], [row[0,-1,0]], np.atleast_1d(row[0,-1,nl//2::nl//lstep]), np.atleast_1d(row[0,-1,nl//2::-(nl//lstep)]),
                        [row[-1,-1,-1]], [row[-1,-1,0]], np.atleast_1d(row[-1,-1,nl//2::nl//lstep]), np.atleast_1d(row[-1,-1,nl//2::-(nl//lstep)])))
    
    use_row_for_grid = np.zeros(nrows, dtype='bool')
    use_row_for_grid[corners] = True
    interprow = row[nx//2::nx//step,ny//2::ny//step,nl//2::nl//lstep].ravel()
    
    interprow = np.append(interprow, row[nx//2::-(nx//step),ny//2::-(ny//step),nl//2::-(nl//lstep)].ravel())
    
    internalpts = np.zeros(nrows, dtype='bool')
    internalpts[interprow] = True

    if flip:
        internalpts = np.invert(internalpts)
    use_row_for_grid |= internalpts
    #use_row_for_grid &= (db[:,-1]>0)
    print(step, lstep, flip)
    print(np.sum(internalpts)/internalpts.size)
     
    if not np.any(use_row_for_grid):
        raise ValueError()
    print(np.sum(use_row_for_grid)/use_row_for_grid.size)        
    return use_row_for_grid
 

#7.12.19 old getoutdata, replaced by gridthin stuffs
def getoutdata(db,step,lstep,flip):
#set up arrays for OUT (& convert to pixels)
    use_row_for_test = np.invert(gridrows(db,step,lstep,flip))
    outxf = db[use_row_for_test,0]
    outyf = db[use_row_for_test,1]
    outlam = db[use_row_for_test,2]
    outxc = db[use_row_for_test,3]/0.015
    outyc = db[use_row_for_test,4]/0.015
    
    out_c = np.array([outxc.ravel(), outyc.ravel()]).T
    
    return outxf, outyf, outlam, outxc, outyc, out_c

#7.23.19
"""     
class WFOSBlueFocalPlane2Detector(FocalPlane2Detector):
    def __init__(self,whatfile,step=None):
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
"""   
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

#7.26.19 failed tricubic testing
    #tricubicinterp goes in WFOS baseclass after interp
    def tricubicinterp(self, outxf, outyf, outlam): 
        if self.interptri is None:
            self.interptri = tricubic(np.append(self.f, np.hstack((self.c, np.random.normal(size = self.c.shape[0]).reshape(-1,1))), axis = 1))
        
        xi = np.array([outxf.ravel(), outyf.ravel(), outlam.ravel()]).T

        return self.interptri.Query(xi)[0]
    class WFOSF2DTricubic(FocalPlane2Detector):
     def __init__(self,whatfile,nx,ny,nl):
        modelfile = os.path.join(os.environ['ENYO_DIR'], 'data', 'instr_models', 'wfos', whatfile)
        self.db = np.genfromtxt(modelfile)
        
        nrows = nx*ny*nl
        row = np.arange(nrows).reshape(nx,ny,nl)

        self.use_row_for_grid = np.ones(row.size, dtype=bool) 
        
        self.use_row_for_grid &= (self.db[:,-1] > 0)
        
        inxf, inyf, inlam, self.inxc, self.inyc = grabdata(self.db,self.use_row_for_grid)
        
        super(WFOSF2DTricubic,self).__init__(inxf, inyf, inlam, self.inxc, self.inyc)
        self.gridfrac = self.gridsize/self.db.shape[0]

    
    
def tricubictest():
    nfield = 1
    random_x = np.random.uniform(-4.2,4.2,size=nfield)
    random_y = np.random.uniform(-1.5,1.5,size=nfield)
    outlam = np.arange(3100, 10000, 10).astype(float)/10000
    fpinterp = WFOSFocalPlane2Detector('Blue_Low_Res_Spot_Data_9x9F_90W.txt',9,9,93) 
    nlam = outlam.size
    savec = np.zeros((nlam*nfield,3), dtype = float)
    
    for i,(x,y) in enumerate(zip(random_x, random_y)):
        outxf = np.full(outlam.size, x, dtype=float)
        outyf = np.full(outlam.size, y, dtype=float)
        outc = fpinterp.tricubicinterp(outxf, outyf, outlam)
        savec[i*nlam:(i+1)*nlam,:2] = outc
        savec[i*nlam:(i+1)*nlam,2] = i
        #plt.scatter(outc[:,0],outc[:,1], c = cm.rainbow(outlam), s = 2)
    print(savec)