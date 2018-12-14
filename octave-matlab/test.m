function [IM, TP, FP, FN, R, P, Q, F1] = test
    % Loading image toolbox - for Octave only, comment out if in Matlab
    pkg load image 
    
    % Read in the two frames for the test
    im1 = imread('../test/diplo000000-L.png'); 
    im2 = imread('../test/diplo000001-L.png');
    
    % Read in the ground truth for the second frame
    [xyroad, xyoccl, gt2] = read_labeled_data('../test/diplo000001-L.txt');
    
    % Perform road detection on the second frame
    [det2, probabilities, labels] = road_detection(im1, im2);
    
    % Resize the result to the original image size 
    det2 = imresize(det2, size(gt2), 'nearest');
    
    % Calculate performance metrics
    [TP, FP, FN, R, P, Q, F1, IM] = calculate_statistics(det2, gt2);
    
    % Announce results
    fprintf('True Positives = %f \n', TP)
    fprintf('False Positives = %f \n', FP)
    fprintf('False Negatives = %f \n', FN)
    fprintf('Precision = %f \n', P)
    fprintf('Recall = %f \n', R)
    fprintf('Quality = %f \n', Q)
    fprintf('F1 score = %f \n', F1)
    
end
