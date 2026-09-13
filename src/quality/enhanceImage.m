function J = enhanceImage(I, opts)
%ENHANCEIMAGE Conservative local contrast enhancement without global colour remapping.
% Per-image illumination normalization is bounded to prevent acquisition-site
% brightness from becoming a shortcut feature (the prior aggressive Ben Graham
% preprocessing showed this failure mode and is deliberately not reproduced).
arguments
    I {mustBeNumeric}; opts.ClipLimit (1,1) double = .012
    opts.BackgroundSigma (1,1) double = 35
end
I = im2uint8(I);
if ndims(I) == 2, lab = cat(3,I,I,I); else, lab = I; end
lab = rgb2lab(lab); L = lab(:,:,1)/100;
background = imgaussfilt(L,opts.BackgroundSigma);
normalized = mat2gray(L - background + median(background(:)));
normalized = imbilatfilt(normalized,.08,5);
lab(:,:,1) = 100*adapthisteq(normalized,'ClipLimit',opts.ClipLimit,'NumTiles',[8 8]);
J = lab2rgb(lab,'OutputType','uint8');
end
