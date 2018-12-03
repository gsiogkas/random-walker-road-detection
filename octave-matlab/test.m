function [TP, FP, FN, R, P, Q, F1, IM] = test

    %load pkg image % For Octave only
    im1 = imread('../test/diplo000000-L.png'); 
    im2 = imread('../test/diplo000001-L.png');
    [xyroad, xyoccl, gt2] = read_labeled_data('../test/diplo000001-L.txt');
    [det2, probabilities, labels] = road_detection(im1, im2);
    det2 = imresize(det2, size(gt2), 'nearest');
    [TP, FP, FN, R, P, Q, F1, IM] = calculate_statistics(det2, gt2);
    fprintf('True Positives = %f \n', TP)
    fprintf('False Positives = %f \n', FP)
    fprintf('False Negatives = %f \n', FN)
    fprintf('Precision = %f \n', P)
    fprintf('Recall = %f \n', R)
    fprintf('Quality = %f \n', Q)
    fprintf('F1 score = %f \n', F1)
end
