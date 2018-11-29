from matplotlib import pyplot as plt
import numpy as np
from matlab_ports import poly2mask


def read_road_annotations(filename, shape):
    f = open(filename)
    lines = f.readlines()
    x = []
    y = []
    for line in lines:
        line = line.replace('\n', '')
        contents = line.split(' ')
        if contents[0] == 'road':
            for i, sample in enumerate(contents[2:]):
                if i % 2:
                    x.append(int (float(sample) * shape[0]) )
                else:
                    y.append(int (float(sample) * shape[1]))
    return x, y


def read_diplodoc(base_path, name, frame_idx):
    im_fnm = base_path +  name + str(frame_idx).zfill(6) + '-L.png'
    im_rgb = plt.imread(im_fnm)
    road_x, road_y = read_road_annotations(im_fnm.replace('png', 'txt'),
                                           np.shape(im_rgb)[:2])
    road_mask = poly2mask(road_x, road_y, np.shape(im_rgb)[:2])
    
    road_mask[-1, :] = True
    for i in np.where(road_mask[:,-2]==True)[0]:
        road_mask[i, -1] = True 
    return im_rgb, road_mask

