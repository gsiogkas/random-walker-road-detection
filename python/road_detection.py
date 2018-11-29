import matplotlib.pyplot as plt
import pyoptflow
from skimage.segmentation import random_walker
from skimage.filters import threshold_otsu
from matlab_ports import imresize
import numpy as np
from skimage.draw import polygon_perimeter
from validation import *
from data_helpers import read_diplodoc


def get_road_seeds(mask_shape, hsc1, thresh, road_trapezoid):
    """Function that produces the road seeds for the random walker algorithm
    
    Arguments:
        mask_shape {tuple (int, int)} -- Shape of image
        hsc1 {[type]} -- [description]
        thresh {[type]} -- [description]
        road_trapezoid {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    rows, cols = mask_shape[0], mask_shape[1]
    road_perim = np.zeros(mask_shape, np.bool)
    rr, cc = polygon_perimeter(road_trapezoid[0], road_trapezoid[1],
                               (rows, cols), clip=True)
    road_perim[rr, cc] = True
    road = np.bitwise_and(road_perim, hsc1 < thresh)
    return road


def get_seeds(hsc1, thresh):
    """Function that generates seeds for the random walker algorithm
    
    Arguments:
        hsc1 {numpy.ndarray(double)} -- HSC1 flow (see paper)
        thresh {float} -- Otsu's threshold for HSC1 flow
    
    Returns:
        numpy.ndarray(uint8) -- seeds for random walker (0: undefined,
                                                         1: road,
                                                         2: non_road)
    """

    sz = np.shape(hsc1)
    rows, cols = sz[0], sz[1]
    top_left = [.7 * rows, .35*cols]
    top_right = [.7 * rows, .65*cols]
    bottom_left = [1 * rows, .1*cols]
    bottom_right = [1 * rows, .9*cols]
    road_trapezoid = [[rows, top_left[0], top_right[0], rows], 
                      [bottom_left[1], top_left[1],
                       top_right[1], bottom_right[1]]
                     ]
    
    # Prepare road seeds (road trapezoid pixels with low HSC1 value)
    road_seeds = get_road_seeds((rows, cols), hsc1, thresh, road_trapezoid)
    
    # Prepare non-road seeds (top 20% fo rows and pixels with high HSC1 value)
    sky = np.zeros_like(road_seeds)
    sky[ :-int(np.ceil(rows * .8)), :] = True
    non_road_seeds = np.bitwise_or(sky, hsc1 >= thresh)

    seeds = np.zeros(hsc1.shape[:2], np.int)
    seeds[road_seeds] = 1
    seeds[non_road_seeds] = 2 

    return seeds


def rgb2c1(RGB):
    """Function that calculates the C1 channel of C1C2C3 colourspace, given a
       RGB image.
    
    Arguments:
        RGB {numpy.ndarray(double)} -- original RGB image
    
    Returns:
        numpy.ndarray(double) -- C1 channel
    """
    assert
    return np.arctan2(RGB[:, :, 0], np.maximum(RGB[:, :, 1], RGB[:, :, 2]))


def main(im1, im2, alpha=3, niter=1, beta=90):
    """Function that takes two consecutive frames from a driving sequence and
       performs road detection based on the paper:

       G. K. Siogkas and E. S. Dermatas, "Random-Walker Monocular Road 
       Detection in Adverse Conditions Using Automated Spatiotemporal Seed 
       Selection," in IEEE Transactions on Intelligent Transportation Systems, 
       vol. 14, no. 2, pp. 527-538, June 2013.

       doi: 10.1109/TITS.2012.2223686

       URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6338335&isnumber=6521414 
    
    Arguments:
        im1 {np.array} -- First frame
        im2 {np.array} -- Second frame
    
    Keyword Arguments:
        alpha {int} -- alpha value for Horn-Schunck calculation (default: {3})
        niter {int} -- iterations for Horn-Schunck calculation (default: {1})
        beta {int} -- beta value for Random Walker calculation (default: {90})
    
    Returns:
        numpy.ndarray(uint8) -- Road detection result
        numpy.ndarray(uint8) -- Random Walker seeds
        
    """

    # Converting original frames to C1
    im1_c1 = rgb2c1(im1)
    im2_c1 = rgb2c1(im2)

    # Downsampling image for segmentation
    im2 = imresize(im2, (120, 160))

    # Calculating the HSC1 flow using Horn Schunck algorithm
    [u, v] = pyoptflow.HornSchunck(im1_c1, im2_c1, alpha, niter)
    hsc1 = imresize(np.sqrt(u ** 2 + v ** 2), np.shape(im2)[:2])
    
    # Getting the threshold of the HSC1 flow using Otsu's method
    threshold = threshold_otsu(hsc1)

    # We use hsc1 to define the road and non-road seeds
    seeds = get_seeds(hsc1, threshold)
    
    # Applying the random walker segmentation on the downsampled image
    result = random_walker(im2, 
                           seeds,
                           multichannel=True, 
                           beta=beta,
                           mode='bf')
    # Upsampling the result to the original size:
    result = imresize(result, np.shape(im2_c1), method='nearest')  

    return result == 1, imresize(seeds, (120, 160), anti_aliasing=False)


def test():
    """Function that tests functionality of road detection module, using two 
       images form the DIPLODOC sequence (see paper).
    
    Returns:
        dict -- Dictionary with performance metrics
        numpy.ndarray(uint8) -- Colour-coded result (green: TP,
                                                     red: FP, 
                                                     yellow: FN)
    """

    im0 = plt.imread('../test/diplo000000-L.png')
    im1, gt1 = read_diplodoc('../test/', 'diplo', 1)
    result1, _ = main(im0, im1)
    results, mask = calculate_metrics(result1, gt1)
    announce_results(results)
    return results, mask


if __name__ == '__main__':
    # Reading frames from disk
    results, mask = test()
    

    
    