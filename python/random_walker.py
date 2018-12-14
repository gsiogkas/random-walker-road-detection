import numpy as np
from scipy import sparse, ndimage as ndi
import numpy as np
import matlab_ports
# from scipy.sparse.linalg import spsolve

try:
    from scikits import umfpack
    old_del = umfpack.UmfpackContext.__del__

    def new_del(self):
        try:
            old_del(self)
        except AttributeError:
            pass
    umfpack.UmfpackContext.__del__ = new_del
    UmfpackContext = umfpack.UmfpackContext()
except:
    UmfpackContext = None

try:
    from pyamg import ruge_stuben_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False

try:
    from sksparse.cholmod import cholesky
    cl_loaded = True
except ImportError:
    cl_loaded = False

from scipy.sparse.linalg import cg
import scipy
from distutils.version import LooseVersion as Version
import functools

if Version(scipy.__version__) >= Version('1.1'):
    cg = functools.partial(cg, atol=0)

def laplacian(edges, weights):
    rows = np.hstack((edges[:, 0], edges[:, 1]))
    cols = np.hstack((edges[:, 1], edges[:, 0]))
    vals = np.hstack((weights, weights))
    N = np.max(rows) + 1
    W = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64)
    L = sparse.dia_matrix((np.reshape(W.sum(axis = 0), (1, N), order='F'), [0]),
                          shape=(N, N), dtype=np.float64) - W
    return L


def lattice(X, Y):  

    # Generate points
    x, y = np.meshgrid(np.arange(Y), np.arange(X))
    points = np.vstack((np.ravel(x, order='F'), np.ravel(y, order='F'))).T
    N = X * Y
    # Connect points
    edges = np.vstack((np.arange(N).T, np.arange(N).T + 1)).T
    edges = np.vstack((np.hstack((edges[:, 0], np.arange(N))), 
                       np.hstack((edges[:, 1], np.arange(N) + X)))).T 
                      
    # Finding lines to exclude
    i, j = np.where(edges > N - 1)
    excluded = np.hstack((i, np.arange(X, X * Y, X)))
    # Removing excluded lines
    edges = edges[~np.in1d(np.arange(edges.shape[0]), excluded)]
    return points, edges


def normalize_weights(weights):
    pass
    
    
def make_weights(edges, vals, val_scale, epsilon=1.e-5):
    vals = vals.astype(np.float64)
    if val_scale > 0:
        val_distances = np.sqrt(np.sum((vals[edges[:, 0], :]
                                        - vals[edges[:, 1], :]) ** 2, axis=1))
        # val_distances = matlab_ports.mat2gray(val_distances)
    else:
        val_distances = np.zeros((np.shape(edges)[0], 1))
        val_scale = 0.0
    weights = np.exp(- val_scale * val_distances) + epsilon
    return weights


def dirichlet_boundary(L, index, vals, method='cl'):
    N = np.shape(L)[0]
    anti_index = np.arange(N)
    anti_index = anti_index[~np.in1d(np.arange(N), index)]
    
    b = - L[anti_index, :][:, index] * (vals)
    b = [sparse.csr_matrix(b[:, 0]), sparse.csr_matrix(b[:, 1])]
    
    L = sparse.csc_matrix(L[anti_index, :][:, anti_index])
    if method == 'cl':
        X = _solve_cl(L, b)
    if method == 'bf':
        X = _solve_bf(L, b)
    if method == 'cg':
        X = _solve_cg(L, b, tol=1.e-3)
    if method == 'cg_mg':
        X = _solve_cg_mg(L, b, tol=1.e-3)
    newVals = np.zeros((N, 2))
    newVals[index, :] = 1 - vals
    newVals[anti_index, :] = X
    return newVals


def spy_sparse(L):
    from matplotlib.pyplot import figure, show
    fig = figure()
    ax1 = fig.add_subplot(111)
    s = str(L.nnz)
    ax1.set_title('nz: ' + s)
    ax1.spy(L, precision=1, markersize=1)
    show()


def _solve_cl(lap_sparse, B, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i using Cholesky decomposition. 
    For each pixel, the label i corresponding to the maximal X_i is returned.
    """
    # B = sparse.csr_matrix(B, shape=lap_sparse.get_shape())
    # lap_sparse = lap_sparse.tocsc()
    solver = cholesky(lap_sparse)
    X = np.array([solver(np.array((-B[i]).todense()).ravel())
                  for i in range(len(B))])
    # if not return_full_prob:
    #     X = np.argmax(X, axis=0)
    return X.T


def _solve_bf(lap_sparse, B, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i for each phase i. An LU decomposition
    of lap_sparse is computed first. For each pixel, the label i
    corresponding to the maximal X_i is returned.
    """
    # lap_sparse = lap_sparse.tocsc()
    solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
    X = np.array([solver(np.array((-B[i]).todense()).ravel())
                  for i in range(len(B))])
 
    # if not return_full_prob:
    #     X = np.argmax(X, axis=0)
    return X.T


def _solve_cg(lap_sparse, B, tol, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i for each phase i, using the conjugate
    gradient method. For each pixel, the label i corresponding to the
    maximal X_i is returned.
    """
    # lap_sparse = lap_sparse.tocsc()
    X = []
    for i in range(len(B)):
        x0 = cg(lap_sparse, -B[i].todense(), tol=tol)[0]
        X.append(x0)
    # if not return_full_prob:
    #     X = np.array(X)
    #     X = np.argmax(X, axis=0)
    return X.T


def _solve_cg_mg(lap_sparse, B, tol, return_full_prob=False):
    """
    solves lap_sparse X_i = B_i for each phase i, using the conjugate
    gradient method with a multigrid preconditioner (ruge-stuben from
    pyamg). For each pixel, the label i corresponding to the maximal
    X_i is returned.
    """
    X = []
    ml = ruge_stuben_solver(lap_sparse)
    M = ml.aspreconditioner(cycle='V')
    for i in range(len(B)):
        x0 = cg(lap_sparse, -B[i].todense(), tol=tol, M=M, maxiter=30)[0]
        X.append(x0)
    if not return_full_prob:
        X = np.array(X)
        X = np.argmax(X, axis=0)
    return X.T


def _clean_labels_ar(X, labels, copy=False):
    X = X.astype(labels.dtype)
    if copy:
        labels = np.copy(labels)
    labels = np.ravel(labels, order='F')
    labels[labels == 0] = X
    return labels


def random_walker(data, labels, beta=90., mode='bf', tol=1.e-3, copy=True,
                  multichannel=False, return_full_prob=False, spacing=None):
    """Random walker algorithm for segmentation from markers.

    Random walker algorithm is implemented for gray-level or multichannel
    images.

    Parameters
    ----------
    data : array_like
        Image to be segmented in phases. Gray-level `data` can be two- or
        three-dimensional; multichannel data can be three- or four-
        dimensional (multichannel=True) with the highest dimension denoting
        channels. Data spacing is assumed isotropic unless the `spacing`
        keyword argument is used.
    labels : array of ints, of same shape as `data` without channels dimension
        Array of seed markers labeled with different positive integers
        for different phases. Zero-labeled pixels are unlabeled pixels.
        Negative labels correspond to inactive pixels that are not taken
        into account (they are removed from the graph). If labels are not
        consecutive integers, the labels array will be transformed so that
        labels are consecutive. In the multichannel case, `labels` should have
        the same shape as a single channel of `data`, i.e. without the final
        dimension denoting channels.
    beta : float
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).
    mode : string, available options {'cg_mg', 'cg', 'bf'}
        Mode for solving the linear system in the random walker algorithm.
        If no preference given, automatically attempt to use the fastest
        option available ('cg_mg' from pyamg >> 'cg' with UMFPACK > 'bf').

        - 'bf' (brute force): an LU factorization of the Laplacian is
          computed. This is fast for small images (<1024x1024), but very slow
          and memory-intensive for large images (e.g., 3-D volumes).
        - 'cg' (conjugate gradient): the linear system is solved iteratively
          using the Conjugate Gradient method from scipy.sparse.linalg. This is
          less memory-consuming than the brute force method for large images,
          but it is quite slow.
        - 'cg_mg' (conjugate gradient with multigrid preconditioner): a
          preconditioner is computed using a multigrid solver, then the
          solution is computed with the Conjugate Gradient method.  This mode
          requires that the pyamg module (http://pyamg.org/) is
          installed. For images of size > 512x512, this is the recommended
          (fastest) mode.
        - 'ch' (Cholesky decomposition based solver) This mode
          requires that the scikit-sparse module 
          (https://pythonhosted.org/scikits.sparse/overview.html#)

    tol : float
        tolerance to achieve when solving the linear system, in
        cg' and 'cg_mg' modes.
    copy : bool
        If copy is False, the `labels` array will be overwritten with
        the result of the segmentation. Use copy=False if you want to
        save on memory.
    multichannel : bool, default False
        If True, input data is parsed as multichannel data (see 'data' above
        for proper input format in this case)
    return_full_prob : bool, default False
        If True, the probability that a pixel belongs to each of the labels
        will be returned, instead of only the most likely label.
    spacing : iterable of floats
        Spacing between voxels in each spatial dimension. If `None`, then
        the spacing between pixels/voxels in each dimension is assumed 1.

    Returns
    -------
    output : ndarray
        * If `return_full_prob` is False, array of ints of same shape as
          `data`, in which each pixel has been labeled according to the marker
          that reached the pixel first by anisotropic diffusion.
        * If `return_full_prob` is True, array of floats of shape
          `(nlabels, data.shape)`. `output[label_nb, i, j]` is the probability
          that label `label_nb` reaches the pixel `(i, j)` first.

    See also
    --------
    skimage.morphology.watershed: watershed segmentation
        A segmentation algorithm based on mathematical morphology
        and "flooding" of regions from markers.

    Notes
    -----
    Multichannel inputs are scaled with all channel data combined. Ensure all
    channels are separately normalized prior to running this algorithm.

    The `spacing` argument is specifically for anisotropic datasets, where
    data points are spaced differently in one or more spatial dimensions.
    Anisotropic data is commonly encountered in medical imaging.

    The algorithm was first proposed in *Random walks for image
    segmentation*, Leo Grady, IEEE Trans Pattern Anal Mach Intell.
    2006 Nov;28(11):1768-83.

    The algorithm solves the diffusion equation at infinite times for
    sources placed on markers of each phase in turn. A pixel is labeled with
    the phase that has the greatest probability to diffuse first to the pixel.

    The diffusion equation is solved by minimizing x.T L x for each phase,
    where L is the Laplacian of the weighted graph of the image, and x is
    the probability that a marker of the given phase arrives first at a pixel
    by diffusion (x=1 on markers of the phase, x=0 on the other markers, and
    the other coefficients are looked for). Each pixel is attributed the label
    for which it has a maximal value of x. The Laplacian L of the image
    is defined as:

       - L_ii = d_i, the number of neighbors of pixel i (the degree of i)
       - L_ij = -w_ij if i and j are adjacent pixels

    The weight w_ij is a decreasing function of the norm of the local gradient.
    This ensures that diffusion is easier between pixels of similar values.

    When the Laplacian is decomposed into blocks of marked and unmarked
    pixels::

        L = M B.T
            B A

    with first indices corresponding to marked pixels, and then to unmarked
    pixels, minimizing x.T L x for one phase amount to solving::

        A x = - B x_m

    where x_m = 1 on markers of the given phase, and 0 on other markers.
    This linear system is solved in the algorithm using a direct method for
    small images, and an iterative method for larger images.

    Examples
    --------
    >>> np.random.seed(0)
    >>> a = np.zeros((10, 10)) + 0.2 * np.random.rand(10, 10)
    >>> a[5:8, 5:8] += 1
    >>> b = np.zeros_like(a)
    >>> b[3, 3] = 1  # Marker for first phase
    >>> b[6, 6] = 2  # Marker for second phase
    >>> random_walker(a, b)
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)

    """
    if mode is None:
        if cl_loaded:
            mode = 'cl'
        elif amg_loaded:
            mode = 'cg_mg'
        elif UmfpackContext is not None:
            mode = 'cg'
        else:
            mode = 'bf'
    elif mode not in ('cg_mg', 'cg', 'bf', 'cl') :
        raise ValueError("{mode} is not a valid mode. Valid modes are 'cg_mg',"
                         " 'cg', 'cl' and 'bf'".format(mode=mode))

    if (labels != 0).all():
        if return_full_prob:
            # Find and iterate over valid labels
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels > 0]

            out_labels = np.empty(labels.shape + (len(unique_labels),),
                                  dtype=np.bool)
            for n, i in enumerate(unique_labels):
                out_labels[..., n] = (labels == i)

        else:
            out_labels = labels
        return out_labels
    

    data = data.astype(np.float64)
    dims = data[..., 0].shape
    # Build graph
    points, edges = lattice(np.shape(data)[0], 
                            np.shape(data)[1])

    if (data.ndim == 3): 
        img_vals = np.vstack((np.ravel(data[:,:,0], order='F'),
                              np.ravel(data[:,:,1], order='F'),
                              np.ravel(data[:,:,2], order='F')))
    elif(data.ndim > 3 | data.ndim == 2): 
        # imgVals = zeros(X*Y,Z);
        # tmp = zeros(X,Y);
        # for zind = 1:Z
        #     tmp = img(:,:,zind);
        #     imgVals(:,zind) = tmp(:);
        # end
        print('Dimensionality of > 3 not yet supported.')
        pass
    else:
        img_vals = np.ravel(data, order='F')
    
    weights = make_weights(edges, img_vals.T, beta)
    L = laplacian(edges, weights)
    road_seeds_ind = np.ravel_multi_index(np.where(labels == 1),
                                          np.shape(labels), order='F')
    non_road_seeds_ind = np.ravel_multi_index(np.where(labels == 2),
                                              np.shape(labels), order='F')
    seeds = np.hstack((road_seeds_ind, non_road_seeds_ind))

    # Solving just for the use case we have, which is bimodal
    boundary = np.zeros((np.size(seeds), 2))

    boundary[:len(road_seeds_ind), 0] = 1
    boundary[len(road_seeds_ind):, 1] = 1
    prob = dirichlet_boundary(L, seeds, boundary, mode)
    prob = np.argmax(prob, axis=1)
    # prob = _clean_labels_ar(prob, labels)
    
    result = np.reshape(prob, dims[:2], order='F')
   
    return result