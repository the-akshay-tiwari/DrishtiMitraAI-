function [probMap, binaryMap] = predictAttentionUNet(net, image, patchSize, stride, threshold)
%PREDICTATTENTIONUNET Sliding-window inference with Gaussian window blending.
%
%   [probMap, binaryMap] = predictAttentionUNet(net, image, patchSize, stride, threshold)
%   - net: trained dlnetwork
%   - image: RGB fundus image (uint8 or single/double)
%   - patchSize: [H, W], default [512, 512]
%   - stride: [sH, sW], default [256, 256] (50% overlap)
%   - threshold: scalar or 1x5 vector, default 0.5

if nargin < 3 || isempty(patchSize)
    patchSize = [512, 512];
end
if nargin < 4 || isempty(stride)
    stride = [256, 256];
end
if nargin < 5 || isempty(threshold)
    threshold = 0.5;
end

[imgH, imgW, ~] = size(image);
pH = patchSize(1);
pW = patchSize(2);

if isinteger(image)
    image = single(image) / 255.0;
else
    image = single(image);
    if max(image(:)) > 1.0
        image = image / 255.0;
    end
end

% Construct 2D Gaussian weighting matrix for seamless patch overlap blending
[Xg, Yg] = meshgrid(linspace(-1, 1, pW), linspace(-1, 1, pH));
w2d = single(exp(-(Xg.^2 + Yg.^2) / 0.5));
w2d = w2d / max(w2d(:));
w5d = repmat(w2d, [1, 1, 5]);

probAccum = zeros(imgH, imgW, 5, 'single');
weightAccum = zeros(imgH, imgW, 5, 'single');

rowStarts = 1:stride(1):(imgH - pH + 1);
if rowStarts(end) ~= (imgH - pH + 1)
    rowStarts = [rowStarts, imgH - pH + 1];
end

colStarts = 1:stride(2):(imgW - pW + 1);
if colStarts(end) ~= (imgW - pW + 1)
    colStarts = [colStarts, imgW - pW + 1];
end

hasGPU = canUseGPU();

for r = rowStarts
    for c = colStarts
        rEnd = r + pH - 1;
        cEnd = c + pW - 1;
        
        patchImg = image(r:rEnd, c:cEnd, :);
        X = dlarray(reshape(patchImg, [pH, pW, 3, 1]), 'SSCB');
        if hasGPU
            X = gpuArray(X);
        end
        
        Y_pred = predict(net, X);
        predData = extractdata(Y_pred);
        
        probAccum(r:rEnd, c:cEnd, :) = probAccum(r:rEnd, c:cEnd, :) + predData .* w5d;
        weightAccum(r:rEnd, c:cEnd, :) = weightAccum(r:rEnd, c:cEnd, :) + w5d;
    end
end

probMap = probAccum ./ max(eps('single'), weightAccum);

if numel(threshold) == 1
    binaryMap = probMap >= threshold;
else
    binaryMap = false(size(probMap));
    for k = 1:5
        binaryMap(:,:,k) = probMap(:,:,k) >= threshold(k);
    end
end
end
