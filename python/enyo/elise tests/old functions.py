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

