function [mask, probabilities, labels] = road_detection(im1, im2,...
                                                        alpha, niter,...
                                                        beta,...
                                                        downsample_sz,...
                                                        previous_road)
    if (~exist('previous_road', 'var'))
       previous_road = [];
    end
    if (~exist('alpha', 'var'))
       alpha = 3;
    end
    if (~exist('niter', 'var'))
        niter = 1;
    end
    if (~exist('beta', 'var'))
        beta = 90;
    end
    if (~exist('downsample_sz', 'var'))
        downsample_sz = [120, 160];
    end

    [u, v] = OpticalFlowOctave(cat(3, rgb2c1(im1), rgb2c1(im2)), alpha, niter);
    hsc1 = sqrt(u .^ 2 + v .^ 2);
    hsc1 = imresize(mat2gray(hsc1), downsample_sz);
    th = graythresh(hsc1);
    
    [non_road_seeds, road_seeds] = get_seeds(hsc1, th, previous_road);
    
    seeds = [road_seeds; non_road_seeds];
    labels = [ones(size(road_seeds)); 2 * ones(size(non_road_seeds))];
    addpath('./graph_functions')
    [mask, probabilities] = random_walker(imresize(im2, downsample_sz),... 
                                          uint32(seeds), labels,...
                                          0, 0, beta);
    mask = 2 - mask;
    mask(:, 1) = mask(:, 2);
    mask(:, end) = mask(:, end - 1);
    mask(end, :) = mask(end - 1, :);
    labels = (hsc1 >= th);
end