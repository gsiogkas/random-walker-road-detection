function [output,bw] = read_diplodoc(frame)

    base = '../data/gtseq/diplo';
    filename = strcat(base, dec2base(frame, 10, 6), '-L.png');
    output(:,:,:,1) = imread(filename);
    mask_filename = strrep(filename, 'png', 'txt');
    [xyroad, xyoccl, bw] = read_labeled_data(mask_filename);