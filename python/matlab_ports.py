import numpy as np
from skimage import draw
from skimage.transform import resize

# Matlab's poly2mask port 
def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """"Convert ROI defined from the row and column coordinates of a polygon's 
        vertices to a binary mask of a given shape.
    
    Arguments:
        vertex_row_coords {list, int} -- Row coordinates of ROI polygon
        vertex_col_coords {list, int} -- Column cordinates of ROI polygon
        shape {tuple, (int, int)} -- Shape of output mask (rows, columns)
    
    Returns:
        numpy.ndarray -- Output mask
    """

    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords,
                                                    vertex_col_coords,
                                                    shape)
    mask = np.full(shape, False)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

# Matlab's mat2gray port (default behaviour)
def mat2gray(A):
    """ Default behaviour of Matlab's mat2gray function.
        I = mat2gray(A) sets the minimum and maximum values of A to 0.0 and 1.0
        and normalizes the values inbetween.
    
    Arguments:
        A {numpy.ndarray} -- The array to be normalized
    
    Returns:
        numpy.ndarray -- The normalized array (range of values in [0, 1])
    """

    A = A.astype(float)
    Amin = np.min(A)
    Amax = np.max(A)
    normalized = (A - Amin) / (Amax - Amin)
    return normalized

# Matlab's imresize port (partial functionality, not guaranteed to be 1:1)
def imresize(im, factor,
             method='bicubic', mode='reflect',
             preserve_range=True, anti_aliasing=True):
    """A partial port of Matlab's imresize. The results are not always the same 
       as Matlab, because of implementation differences of filters for the 
       interpolation. 
    
    Arguments:
        im {numpy.ndarray} -- The array to be resampled
        factor {float, or (int, int)} -- If a constant float, it multiplys both
                                         dimensions by factor. 
                                         If a tuple (int, int), it resizes to 
                                         the dimensions specified by the tuple 
                                         (rows, columns)
    
    Keyword Arguments:
        method {str} -- Interpolation method (default: {'bicubic'})
        mode {str} -- Border handling (default: {'reflect'})
        preserve_range {bool} -- Preserving range of original (default: {True})
        anti_aliasing {bool} -- Anti-aliasing filters (default: {True})
    
    Returns:
        numpy.ndarray -- The resampled result
    """

    methods_dict = {'nearest': 0,
                    'bilinear': 1,
                    'box': 2,
                    'bicubic': 3}
    im_size = np.shape(im)
    if type(factor) is not tuple:
        factor = (np.ceil(im_size[0] * factor), np.ceil(im_size[1] * factor))
        if factor == 1:
            return im
    return resize(im, factor,
                  order=methods_dict[method],
                  mode=mode,
                  preserve_range=preserve_range)
