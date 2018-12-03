function [seeds,labels] = ConvertPoints2Seeds(imSize,xB,yB,xR,yR,xM,yM)

    seeds = (sub2ind(imSize,xB,yB));
    seeds = unique(seeds);
    s1 = size(seeds);
    vR = (sub2ind(imSize,xR,yR));
    vR = setdiff(vR,seeds);
    seeds = [seeds ; vR];
    labels = [ones(s1) ; 2*ones(size(vR))];
    if nargin == 7
        vM = (sub2ind(imSize,xM,yM));
        vM = setdiff(vM,seeds);
        seeds = [seeds ; vM];
        labels = [labels ; 3*ones(size(vM))];
    end
end
