import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from matlab_ports import poly2mask


def my_hook(t):
    """Function that wraps tqdm instance.
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b {int} --  optional
            Number of blocks transferred so far [default: 1].
        bsize {int} --  optional
            Size of each block (in tqdm units) [default: 1].
        tsize {int} --  optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def read_road_annotations(filename, shape):
    """Function to read road annotations in DIPLODC format 
    
    Arguments:
        filename {str} -- Filename (full path) of annotation file 
        shape {tuple} -- Shape of image to translate the coordinates to
    
    Returns:
        int, int -- row and column coordinates for road polygon definition
    """

    with open(filename) as f:
        lines = f.readlines()
        row = []
        col = []
        for line in lines:
            if 'road' in line:
                line = line.replace('\n', '')
                contents = line.split(' ')
                for i, sample in enumerate(contents[2:]):
                    if i % 2:
                        row.append(int (float(sample) * shape[0]))
                    else:
                        col.append(int (float(sample) * shape[1]))
    return row, col


def read_occl_annotations(filename, shape):
    """Function to read occlusion annotations in DIPLODC format 
    
    Arguments:
        filename {str} -- Filename (full path) of annotation file 
        shape {tuple} -- Shape of image to translate the coordinates to
    
    Returns:
        list of lists (int, int) -- row and column coordinates for occlusions 
                                    polygons definition
    """

    with open(filename) as f:
        lines = f.readlines()
        rows = []
        cols = []
        for line in lines:
            if 'occl' in line:
                row = []
                col = []
                line = line.replace('\n', '')
                contents = line.split(' ')
                for i, sample in enumerate(contents[2:]):
                    if i % 2:
                        row.append(np.ceil (float(sample) * shape[0]))
                    else:
                        col.append(np.ceil (float(sample) * shape[1]))
                rows.append(row)
                cols.append(col)
    return rows, cols


def read_diplodoc(base_path, name='diplo', frame_idx=0):
    """Function to read an image and accompanying annotation from DIPLODOC 
    
    Arguments:
        base_path {str} -- Path where image and annotation live
        name {str} -- Prefix of files [default:'diplo']
        frame_idx {int} -- Index of frame to retrieve [default:0]
    
    Returns:
        numpy.ndarray -- Retrieved RGB image
        numpy.ndarray -- Road mask of same dimensions as the original image
    """

    im_fnm = base_path +  name + str(frame_idx).zfill(6) + '-L.png'
    im_rgb = plt.imread(im_fnm)
    road_x, road_y = read_road_annotations(im_fnm.replace('png', 'txt'),
                                           np.shape(im_rgb)[:2])
    occl_x, occl_y = read_occl_annotations(im_fnm.replace('png', 'txt'),
                                           np.shape(im_rgb)[:2])
    road_mask = poly2mask(road_x, road_y, np.shape(im_rgb)[:2])
    if occl_x:
        occl_mask = np.zeros_like(road_mask)
        for oc_x, oc_y in zip(occl_x, occl_y):
            occl_mask = np.bitwise_or(occl_mask,
                                      poly2mask(oc_x, oc_y,
                                                np.shape(im_rgb)[:2]))
        road_mask = np.bitwise_and(road_mask, np.bitwise_not(occl_mask))
        
    road_mask[-1, :] = True
    for i in np.where(road_mask[:,-2]==True)[0]:
        road_mask[i, -1] = True 
    return im_rgb, road_mask


def download_decompress_diplodoc():
    """Function to download DIPLODOC data if it's not already present inside 
       the ../data/gtseq directory
    """

    final_data_path = Path(Path.cwd()).parents[0] / 'data'
    base_data_path = final_data_path.as_posix()
    if not (final_data_path / 'gtseq').is_dir():
        diplodoc_url = 'https://tev-static.fbk.eu/DATABASES/gtseq002836.tgz'
        print('Getting DIPLODOC dataset -- progress:')
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc='Downloading') as t:
            urllib.request.urlretrieve(diplodoc_url,
                                       reporthook=my_hook(t),
                                       filename=(base_data_path + '/gtseq.tgz'),
                                       data=None)
        # local_filename, headers = urllib.request.urlretrieve(diplodoc_url)
        with tarfile.open(base_data_path + '/gtseq.tgz') as tar_file:
            for tar_member in tqdm(iterable=tar_file.getmembers(),
                                   total=len(tar_file.getmembers()),
                                   desc='Decompressing'):
                tar_file.extract(member=tar_member, path=base_data_path) 
        tar_file.close()
    else:
        print('Directory already exists. Aborting...')

if __name__ == '__main__':
    pass
