function output = rgb2c1(R,G,B)

warning off;

if nargin == 1
    Rgb = double(R);
    R = Rgb(:,:,1);
    G = Rgb(:,:,2);
    B = Rgb(:,:,3);
else
    R = double(R);
    G = double(G);
    B = double(B);
end
 den = max([G(:) B(:)],[],2);
 output = atan2(R(:), den);
 %output = atan(R(:)./den);
 
% output(:,:,2) = atan(RGB(:,:,2)./max(cat(3,RGB(:,:,1),RGB(:,:,3)),[],3));
% output(:,:,3) = atan(RGB(:,:,3)./max(cat(3,RGB(:,:,1),RGB(:,:,2)),[],3));
% den = max([R(:) B(:)],[],2);
% output = atan(G(:)./den);

%output(isnan(output)) = 0;
%output(output==-100) = max(max(output));
%output = output/max(max(output));
output = reshape(output,[size(R,1) size(R,2)]);