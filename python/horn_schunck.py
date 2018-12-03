from scipy.ndimage.filters import convolve as filter2
import numpy as np
#
HSKERN =np.array([[1/12, 1/6, 1/12],
                  [1/6,    0, 1/6],
                  [1/12, 1/6, 1/12]],float)

kernelX = np.array([[-1, 1],
                     [-1, 1]]) * .25 #kernel for computing d/dx

kernelY = np.array([[-1,-1],
                     [ 1, 1]]) * .25 #kernel for computing d/dy

kernelT = np.ones((2,2))*.25


# The following code is adapted from:
# https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
def conv2(x,y,mode='same'):
    """
    Emulate the function conv2 from Mathworks.

    Usage:

    z = conv2(x,y,mode='same')

    TODO: 
     - Support other modes than 'same' (see conv2.m)
    """

    if not(mode == 'same'):
        raise Exception("Mode not supported")

    # Add singleton dimensions
    if (len(x.shape) < len(y.shape)):
        dim = x.shape
        for i in range(len(x.shape),len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)
    elif (len(y.shape) < len(x.shape)):
        dim = y.shape
        for i in range(len(y.shape),len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = ()

    # Apparently, the origin must be set in a special way to reproduce
    # the results of scipy.signal.convolve and Matlab
    for i in range(len(x.shape)):
        if ( (x.shape[i] - y.shape[i]) % 2 == 0 and
             x.shape[i] > 1 and
             y.shape[i] > 1):
            origin = origin + (-1,)
        else:
            origin = origin + (0,)

    z = filter2(x,y, mode='constant', origin=origin)

    return z


def HornSchunck(im1, im2, alpha:float=0.001, Niter:int=8, verbose:bool=False):
    """
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

	#set up initial velocities
    uInitial = np.zeros([im1.shape[0],im1.shape[1]])
    vInitial = np.zeros([im1.shape[0],im1.shape[1]])

	# Set initial value for the flow vectors
    U = uInitial
    V = vInitial

	# Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    if verbose:
        from .plots import plotderiv
        plotderiv(fx,fy,ft)

#    print(fx[100,100],fy[100,100],ft[100,100])

	# Iteration to reduce error
    for _ in range(Niter):
#%% Compute local averages of the flow vectors
        uAvg = conv2(U, HSKERN)
        vAvg = conv2(V, HSKERN)
#%% common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
#%% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U,V


def computeDerivatives(im1, im2):

    fx = conv2(im1,kernelX) + conv2(im2,kernelX)
    fy = conv2(im1,kernelY) + conv2(im2,kernelY)

   # ft = im2 - im1
    ft = conv2(im1,kernelT) + conv2(im2,-kernelT)

    return fx,fy,ft