function [mask, probabilities, labels] = road_detection(im1, im2,...
                                                        alpha, niter,...
                                                        beta, downsample_sz)

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
    road_trapezoid = poly2mask( [floor(.1 * downsample_sz(2)),...  
                                floor(.35 * downsample_sz(2)),...
                                floor(.65 * downsample_sz(2)),...
                                floor(.9 * downsample_sz(2))],...
                                [downsample_sz(1) - 1,...
                                floor(.7 * downsample_sz(1)),...
                                floor(.7 * downsample_sz(1)),...
                                downsample_sz(1) - 1],...
                                downsample_sz(1),...
                                downsample_sz(2));
    road_perim = bwperim(road_trapezoid);
    [rows_road, cols_road] = find(road_perim & hsc1 < th);

    sky = zeros(downsample_sz);
    sky(1:floor(.2 * downsample_sz(1)), :) = 1;
    [rows_non_road, cols_non_road] = find(sky | hsc1 >= th);

    non_road_seeds = (sub2ind(downsample_sz, rows_non_road, cols_non_road));
    road_seeds = (sub2ind(downsample_sz, rows_road, cols_road));
    seeds = [road_seeds; non_road_seeds];
    labels = [ones(size(road_seeds)); 2 * ones(size(non_road_seeds))];
    addpath('./graph_functions')
    [mask,probabilities] = random_walker(imresize(im2, downsample_sz),... 
                                         uint32(seeds), labels,...
                                         0, 0, beta);
    mask = 2 - mask;
    labels = (hsc1 >= th);
end