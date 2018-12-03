# MIT License

# Copcol_roadight (c) 2018 George Siogkas

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copcol_roadight notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPcol_roadIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import pyoptflow
from skimage.segmentation import random_walker
from skimage.filters import threshold_otsu
from matlab_ports import imresize
import numpy as np
from skimage.draw import polygon_perimeter
from validation import *
from data_helpers import read_diplodoc, download_decompress_diplodoc
from pathlib import Path
from tqdm import tqdm, trange
from drawnow import drawnow
from time import time


def adaptive_perimeter_update(previous_road, hsc1_bin):
    """Function to adaptively generate coordinates for road trapezoid
    
    Arguments:
        previous_road {np.ndarray} -- previous frame's detected road
        hsc1_bin {np.ndarray} -- hsc1 flow binarized
    Returns:
        list{int}, list{int} -- row, column coordinates of road trapezoid
    """
    row_road, col_road = np.where(np.bitwise_and(previous_road > 0,  hsc1_bin))
    base = np.max(col_road) - np.min(col_road)
    height = np.max(row_road) - np.min(row_road)
    centroid = (np.mean(row_road), np.mean(col_road))
    sz = np.shape(previous_road)
    rows, cols = sz[0], sz[1]
    top_row = max(np.min(row_road), int(rows * 0.5)) + int(height / 8)
    # top_row = int((centroid[0] + np.min(row_road)) / 2)
    dx = int(np.min(row_road) - centroid[0]) / 2
    yy = np.where(previous_road[top_row, :] == 1)
    min_yy, max_yy = np.min(yy), np.max(yy)
    top_width = max_yy - min_yy
    top_left = [top_row, min_yy + int(0.35 * top_width)]
    top_right = [top_row, max_yy - int(0.35 * top_width)]
    bottom_left = [rows, np.min(col_road) + 0.05 * base]
    bottom_right = [rows, np.min(col_road) + 0.95 * base]

    rr, cc = polygon_perimeter([rows, top_left[0], top_right[0], rows], 
                               [bottom_left[1], top_left[1],
                                top_right[1], bottom_right[1]],
                               (rows, cols), clip=True)
    return rr, cc


def get_road_seeds(mask_shape, hsc1, thresh, road_trapezoid, previous_road=[]):
    """Function that produces the road seeds for the random walker algorithm
    
    Arguments:
        mask_shape {tuple (int, int)} -- Shape of image
        hsc1 {[type]} -- [description]
        thresh {[type]} -- [description]
        road_trapezoid {[type]} -- [description]
        previous_road {list or np.ndarray} -- previous frame's detected road or 
                                              [] if static seed selection
    
    Returns:
        [type] -- [description]
    """

    rows, cols = mask_shape[0], mask_shape[1]
    road_perim = np.zeros(mask_shape, np.bool)
    if len(previous_road):
        rr, cc = adaptive_perimeter_update(previous_road, hsc1 < thresh)
    else:
        rr, cc = polygon_perimeter(road_trapezoid[0], road_trapezoid[1],
                                   (rows, cols), clip=True)
    road_perim[rr, cc] = True
    road = np.bitwise_and(road_perim, hsc1 < thresh)
    return road


def get_seeds(hsc1, thresh, previous_road):
    """Function that generates seeds for the random walker algorithm
    
    Arguments:
        hsc1 {numpy.ndarray(double)} -- HSC1 flow (see paper)
        thresh {float} -- Otsu's threshold for HSC1 flow
        previous_road {list or np.ndarray} -- previous frame's detected road or 
                                              [] if static seed selection

    Returns:
        numpy.ndarray(uint8) -- seeds for random walker (0: undefined,
                                                         1: road,
                                                         2: non_road)
    """

    sz = np.shape(hsc1)
    rows, cols = sz[0], sz[1]
    top_left = [.7 * rows, .35 * cols]
    top_right = [.7 * rows, .65 * cols]
    bottom_left = [rows - 1, .1 * cols]
    bottom_right = [rows - 1, .9 * cols]
    road_trapezoid = [[rows, top_left[0], top_right[0], rows], 
                      [bottom_left[1], top_left[1],
                       top_right[1], bottom_right[1]]
                     ]
    
    # Prepare road seeds (road trapezoid pixels with low HSC1 value)
    road_seeds = get_road_seeds((rows, cols), hsc1, thresh,
                                road_trapezoid, previous_road)
    
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
    return np.arctan2(RGB[:, :, 0], np.maximum(RGB[:, :, 1], RGB[:, :, 2]))


def main(im1, im2, previous_result=[], seed_selection='static',
         alpha=3, niter=1,
         beta=90, rw_method='bf', downsample_sz=(120, 160)):
    """Function that takes two consecutive frames from a driving sequence and
       performs road detection based on the paper:

       G. K. Siogkas and E. S. Dermatas, "Random-Walker Monocular Road 
       Detection in Adverse Conditions Using Automated Spatiotemporal Seed 
       Selection," in IEEE Transactions on Intelligent Transportation Systems, 
       vol. 14, no. 2, pp. 527-538, June 2013.

       doi: 10.1109/TITS.2012.2223686

    URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6338335&isnumber=6521414 
    
    Arguments:
        im1 {np.array} -- first frame
        im2 {np.array} -- second frame
    
    Keyword Arguments:
        alpha {int} -- alpha value for Horn-Schunck calculation (default: {3})
        niter {int} -- iterations for Horn-Schunck calculation (default: {1})
        beta {int} -- beta value for random walker calculation (default: {90})
        rw_method {str} -- method for random walker solver (default: {'bf'})
        downsample_sz {tuple (int, int)} -- downsampling size for random walker 
                                            input image(default: {(120, 160))})
    
    Returns:
        numpy.ndarray(uint8) -- road detection result
        numpy.ndarray(uint8) -- random walker seeds
        
    """

    # Seed selection dict
    ssd = {'static':[], 'adaptive':previous_result}
    
    # Converting original frames to C1
    im1_c1 = rgb2c1(im1)
    im2_c1 = rgb2c1(im2)

    # Downsampling image for segmentation
    im2 = imresize(im2, downsample_sz)

    # Calculating the HSC1 flow using Horn Schunck algorithm
    [u, v] = pyoptflow.HornSchunck(im1_c1, im2_c1, alpha, niter)
    hsc1 = imresize(np.sqrt(u ** 2 + v ** 2), np.shape(im2)[:2])
    # hsc1 = mat2gray(hsc1)
    # Getting the threshold of the HSC1 flow using Otsu's method
    threshold = threshold_otsu(hsc1)

    # We use hsc1 to define the road and non-road seeds
    seeds = get_seeds(hsc1, threshold, ssd[seed_selection])
    
    # Applying the random walker segmentation on the downsampled image
    result = random_walker(im2,
                           seeds,
                           multichannel=True, 
                           beta=beta,
                           mode=rw_method)
    # Upsampling the result to the original size:
    result = imresize(result, np.shape(im2_c1), method='nearest')  

    return result == 1


def test():
    """Function that tests functionality of road detection module, using two 
       images form the DIPLODOC sequence (see paper).
    
    Returns:
        dict -- dictionary with performance metrics
        numpy.ndarray(uint8) -- colour-coded result (green: TP,
                                                     red: FP, 
                                                     yellow: FN)
        numpy.ndarray -- original image used 
    """

    im0 = plt.imread('../test/diplo000000-L.png')
    im1, gt1 = read_diplodoc('../test/', 'diplo', 1)
    result1 = main(im0, im1)
    results, mask = calculate_metrics(result1, gt1)
    announce_results(results)
    return results, mask, im1


def test_on_diplodoc(seed_selection='static',
                     alpha=3, niter=1,
                     beta=90, rw_method='bf',
                     downsample_sz=(120, 160)):
    """Function to test road detection on DIPLODOC sequence
    
    Keyword Arguments:
        alpha {int} -- See main() (default: {3})
        niter {int} -- See main() (default: {1})
        beta {int} -- See main() (default: {90})
        rw_method {str} -- See main() (default: {'bf'})
        downsample_sz {tuple} -- See main() (default: {(120, 160)})
    
    Returns:
        list -- a list of dictionaries with per frame metrics
    """
    previous_result = []
    # Frame indices for start and end of each sub-sequence in DIPLODOC
    seq_frames = [(0, 450), (451, 601), (602, 702), (703, 763), (764, 864)]
    metrics = []
    cnt = 0
    t0 = time()
    for seq in seq_frames:
        # Check if data exists, if not, download first
        diplodoc_data_path = Path(Path.cwd()).parents[0] / 'data' / 'gtseq'
        if not diplodoc_data_path.is_dir():
            download_decompress_diplodoc()
        start_frame = seq[0]
        end_frame = seq[1]
        step = 1
        t = trange(start_frame, end_frame, step,
                   total=end_frame - start_frame,
                   unit="frames",
                   desc="F1=0"
                  )
        for i in t:
            im0, _ = read_diplodoc(diplodoc_data_path.as_posix() + '/',
                                   frame_idx = i)
            im1, gt1 = read_diplodoc(diplodoc_data_path.as_posix() + '/',
                                     frame_idx = i + 1)
            det1 = main(im0, im1, previous_result=previous_result,
                        seed_selection=seed_selection,
                        alpha=alpha, niter=niter,
                        beta=beta, rw_method=rw_method,
                        downsample_sz=downsample_sz)
            
            m, colored = calculate_metrics(det1, gt1)
            t.set_description("F1=" + str(m['F1'])[:4])
            t.refresh()
            cnt += 1
            # plt.imsave(str(cnt) + '.png', colored)
            previous_result = imresize(det1, downsample_sz, method='nearest')
            metrics.append(m)
    t1 = time()
    mrt = t1 - t0 / len(metrics)
    print('Mean runtime: ', mrt, 'seconds (', 1 / mrt, 'fps)')
    return metrics


if __name__ == '__main__':
    # Reading frames from disk
    results, mask, im1 = test()
