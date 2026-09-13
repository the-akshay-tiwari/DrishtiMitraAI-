function quality = assessQuality(I, opts)
%ASSESSQUALITY Measure fundus gradeability and return actionable recapture advice.
arguments
    I {mustBeNumeric}
    opts.MinScore (1,1) double = 0.55
    opts.MinFOV (1,1) double = 0.45
    opts.MinSharpness (1,1) double = 15
end
if ndims(I) == 3, G = rgb2gray(im2uint8(I)); else, G = im2uint8(I); end
Gd = im2double(G);
% Saturation and luminance jointly give a robust retinal field estimate.
if ndims(I) == 3
    hsvI = rgb2hsv(im2double(I)); retinalMask = hsvI(:,:,2) > .08 & hsvI(:,:,3) > .06;
else
    retinalMask = Gd > graythresh(Gd)/2;
end
retinalMask = imfill(bwareafilt(retinalMask,1),'holes');
retinalMask = imopen(retinalMask,strel('disk',5));
fov = nnz(retinalMask)/numel(retinalMask);
L = imfilter(Gd,fspecial('laplacian',0.2),'replicate');
sharpness = var(L(retinalMask),'omitnan');
pixels = Gd(retinalMask);
mu = mean(pixels,'omitnan'); sigma = std(pixels,'omitnan');
under = mean(pixels < .08); over = mean(pixels > .95);
sharpScore = min(1,sharpness/opts.MinSharpness);
lightScore = max(0,1 - 2*abs(mu-.5) - max(0,.18-sigma) - 2*(under+over));
fovScore = min(1,fov/opts.MinFOV);
score = .4*sharpScore + .35*lightScore + .25*fovScore;
reasons = strings(0,1);
if fov < opts.MinFOV, reasons(end+1) = "Move camera closer and centre the retinal field; too little retina is visible."; end
if sharpness < opts.MinSharpness, reasons(end+1) = "Retake after stabilising camera and refocusing; image blur obscures small lesions."; end
if mu < .25 || under > .12, reasons(end+1) = "Increase illumination or exposure; the fundus is underexposed."; end
if mu > .78 || over > .08, reasons(end+1) = "Reduce illumination or exposure; highlights are saturated."; end
if sigma < .12, reasons(end+1) = "Improve even illumination; contrast across the retinal field is inadequate."; end
quality = struct('retinalMask',retinalMask,'fieldOfViewFraction',fov, ...
    'laplacianVariance',sharpness,'luminanceMean',mu,'luminanceStd',sigma, ...
    'underExposureFraction',under,'overExposureFraction',over,'gradeabilityScore',score, ...
    'isGradeable',score >= opts.MinScore && isempty(reasons), ...
    'recaptureReasons',reasons);
end
