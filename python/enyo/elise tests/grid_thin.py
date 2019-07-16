
import warnings
import numpy as np

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


def main():
    # Set the row numbers
    nx = 9
    ny = 9
    nw = 93
    rows = np.arange(nx*ny*nw).reshape(nx,ny,nw)

    # Get the number of iterations to thin
    iters = iteration_steps(rows)

    for i in iters:
        # Get the rows to use in the interpolator
        thinned_rows = grid_thin_indx(rows, i)
        print(i, thinned_rows.size, rows.size, thinned_rows.size/rows.size)

        use_rows_in_grid = np.zeros(rows.size, dtype=bool)
        use_rows_in_grid[thinned_rows] = True

        # ...


if __name__ == '__main__':
    main()    

"""
from matplotlib import pyplot, colors, cm
import numpy as np

nlines = 5
colors = cm.get_cmap('seismic')(np.linspace(0,1,nlines))

x = np.arange(10)
y = np.zeros(10)

for i in range(nlines):
     pyplot.plot(x, y+i, color=colors[i])
pyplot.show()

"""
#1 27 7533 0.0035842293906810036
#2 125 7533 0.016593654586486126
#3 585 7533 0.07765830346475508
#4 747 7533 0.0991636798088411
#5 909 7533 0.12066905615292713
#6 1152 7533 0.15292712066905614
#7 1314 7533 0.17443249701314217
#8 1557 7533 0.2066905615292712
#9 1557 7533 0.2066905615292712
#10 1953 7533 0.25925925925925924
#11 1953 7533 0.25925925925925924
#12 2529 7533 0.33572281959378736
#13 2529 7533 0.33572281959378736
#14 2529 7533 0.33572281959378736
#15 2529 7533 0.33572281959378736
#16 3807 7533 0.5053763440860215
#17 3807 7533 0.5053763440860215
#18 3807 7533 0.5053763440860215
#19 3807 7533 0.5053763440860215
#20 3807 7533 0.5053763440860215
#21 3807 7533 0.5053763440860215
#22 3807 7533 0.5053763440860215
#23 3807 7533 0.5053763440860215
#-23 3753 7533 0.4982078853046595
#-22 3753 7533 0.4982078853046595
#-21 3753 7533 0.4982078853046595
#-20 3753 7533 0.4982078853046595
#-19 3753 7533 0.4982078853046595
#-18 3753 7533 0.4982078853046595
#-17 3753 7533 0.4982078853046595
#-16 3753 7533 0.4982078853046595
#-15 5031 7533 0.6678614097968937
#-14 5031 7533 0.6678614097968937
#-13 5031 7533 0.6678614097968937
#-12 5031 7533 0.6678614097968937
#-11 5607 7533 0.7443249701314217
#-10 5607 7533 0.7443249701314217
#-9 6003 7533 0.7968936678614098
#-8 6003 7533 0.7968936678614098
#-7 6246 7533 0.8291517323775388
#-6 6408 7533 0.8506571087216248
#-5 6651 7533 0.8829151732377539
#-4 6813 7533 0.9044205495818399
#-3 6975 7533 0.9259259259259259
#-2 7435 7533 0.9869905748041948


