import numpy as np
from numpy import count_nonzero as nnz


def calculate_metrics(roadSegmResult, groundTruth):
    """Performance metrics per frame

    Arguments:
        roadSegmResult {bool} -- mask of road segmentation for frame
        groundTruth {bool} -- ground truth mask for frame

    Returns:
        results {dict}:
            TP {int} -- True Positive count 
            FP {int} -- False Positive count 
            FN {int} -- False Negative count
            R {double} -- Recall (Completeness)
            P {double} -- Precision (Correctness) 
            Q {double} -- Quality metric 
            F1 {double} -- F1 score
        mask {np.array(np.uint8)} -- Color coded mask (Green:TP, 
                                                       Red:FP,
                                                       Yellow: FN,
                                                       Black:  TN)
    """

    results = {}
    r = np.zeros_like(roadSegmResult, np.uint8)
    g = np.zeros_like(roadSegmResult, np.uint8)
    b = np.zeros_like(roadSegmResult, np.uint8)
    
    results['TP'] = nnz(np.bitwise_and(roadSegmResult, groundTruth))
    results['FP'] = nnz(np.bitwise_and(roadSegmResult, groundTruth == 0)) 
    results['FN'] = nnz(np.bitwise_and(roadSegmResult == 0 , groundTruth))
    results['R'] = results['TP'] / (results['TP'] + results['FN'])
    results['P'] = results['TP'] / (results['TP'] + results['FP'])
    results['Q']  = results['TP'] / (results['TP']
                                    + results['FN']
                                    + results['FP'])
    results['F1'] = (2 * results['P'] * results['R']) \
                  / (results['P'] + results['R'])

    g[np.bitwise_and(roadSegmResult, groundTruth)] = 255
    r[np.bitwise_and(roadSegmResult, groundTruth == 0)] = 255
    g[np.bitwise_and(roadSegmResult == 0 , groundTruth)] = 255
    r[np.bitwise_and(roadSegmResult == 0 , groundTruth)] = 255
    mask = np.stack((r, g, b), axis=-1)
    
    return results, mask


def announce_results(results, mode='frame'):
    prefix_dict = {'frame':'', 'average':'Average '}
    print(prefix_dict[mode] + 'True Positives  :', results['TP'])
    print(prefix_dict[mode] + 'False Positives :', results['FP'])
    print(prefix_dict[mode] + 'False Negatives :', results['FN'])
    print(prefix_dict[mode] + 'Recall          :', results['R'])
    print(prefix_dict[mode] + 'Precision       :', results['P'])
    print(prefix_dict[mode] + 'Quality         :', results['Q'])
    print(prefix_dict[mode] + 'F1 score        :', results['F1'])