function [statistics] = test_on_diplodoc(alpha, niter, beta, downsample_sz)
  
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
    seq_frames = [[0, 450]; [451, 601]; [602, 702]; [703, 763]; [764, 864]];
    statistics = [];
    more off
    for i = 1:5
        start_frame = seq_frames(i, 1);
        end_frame = seq_frames(i, 2);
        for frame = start_frame:end_frame - 1
            tic;
            [im1, gt1] = read_diplodoc(frame);
            [im2, gt2] = read_diplodoc(frame + 1);
            [mask, probabilities, seeds] = road_detection(im1, im2, ...
                                                          alpha, niter,...
                                                          beta, downsample_sz);
            gt2 = imresize(gt2, size(mask), 'nearest');
            [TP, FP, FN, R, P, Q, F1, I] = calculate_statistics(mask, gt2);
            statistics = cat(1, statistics, [TP, FP, FN, R, P, Q, F1]);
            dt = toc;
            fps = 1 / dt;
            fprintf ('Processed frame %d in %f seconds (%f fps)\n', frame, dt, fps);
        end
    end
    mv = mean(statistics)
    sd = std(statistics)
    fprintf('Average Precision = %f (+/-%f)\n', mv(5), sd(5))
    fprintf('Average Recall = %f (+/-%f)\n', mv(4), sd(4))
    fprintf('Average Quality = %f (+/-%f)\n', mv(6), sd(7))
    fprintf('Average F1 score = %f (+/-%f)\n', mv(7), sd(7))
end