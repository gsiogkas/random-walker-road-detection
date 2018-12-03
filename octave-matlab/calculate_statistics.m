function [TP, FP, FN, R, P, Q, F1, IM] = calculate_statistics(seg_res, gt)
    TP_im = (seg_res & gt) == 1;
    TP = nnz(TP_im);
    FP_im = (seg_res == 1) & (gt == 0);
    FP = nnz(FP_im);
    FN_im = (seg_res == 0) & (gt == 1);
    FN = nnz(FN_im);
    R = TP / (TP + FN);
    P = TP / (TP + FP);
    Q  = TP / (TP + FN + FP);
    F1 = (2 * R * P) / (R + P);
    
    IM = cat(3, FP_im, TP_im, zeros(size(TP_im)));
    IM(:,:,1) = IM(:,:,1) + FN_im;
    IM(:,:,2) = IM(:,:,2) + FN_im;
end
   