function J = preprocessFundusForClassifier(I, inputSizeOrNetwork)
%PREPROCESSFUNDUSFORCLASSIFIER Prepare one grayscale/RGB fundus image for classification.
% Accepted inputs are M-by-N grayscale or M-by-N-by-3 RGB uint8/single/double.
% Floating point inputs must be finite and in [0,1] (normalized) or [0,255]
% (intensity scale); other values are rejected rather than silently assumed.
% The existing enhancement policy is applied once, then the image is resized.
% Output is a deterministic targetHeight-by-targetWidth-by-3 RGB uint8 image.

if nargin < 2 || isempty(inputSizeOrNetwork)
    inputSizeOrNetwork = [224 224 3];
end

validateImage(I);
Iu8 = toUint8(I);
if ndims(Iu8) == 2
    Iu8 = repmat(Iu8,1,1,3);
end

targetSize = resolveInputSize(inputSizeOrNetwork);

Iu8 = imresize(Iu8,targetSize(1:2),'bilinear');

J = enhanceImageV2(Iu8); % V2 enhancement: preserve fine retinal structures.
end

function validateImage(I)
if ~(isnumeric(I) && (isa(I,'uint8') || isa(I,'single') || isa(I,'double')))
    error('DrishtiMitra:PreprocessType', ...
        'Input must be a uint8, single, or double numeric image.');
end
if ndims(I) ~= 2 && ~(ndims(I) == 3 && size(I,3) == 3)
    error('DrishtiMitra:PreprocessDimensions', ...
        'Input must be M-by-N grayscale or M-by-N-by-3 RGB; received size %s.',mat2str(size(I)));
end
end

function Iu8 = toUint8(I)
if isa(I,'uint8')
    Iu8 = I;
    return
end
if any(~isfinite(I),'all')
    error('DrishtiMitra:PreprocessFiniteValues', ...
        'Floating-point input must contain only finite values.');
end
minimum = min(I,[],'all');
maximum = max(I,[],'all');
if minimum >= 0 && maximum <= 1
    Iu8 = im2uint8(I);
elseif minimum >= 0 && maximum <= 255
    Iu8 = uint8(round(I));
else
    error('DrishtiMitra:PreprocessFloatRange', ...
        'Floating-point input must be entirely in [0,1] or [0,255]; observed range [%g, %g].',minimum,maximum);
end
end

function targetSize = resolveInputSize(inputSizeOrNetwork)
if isnumeric(inputSizeOrNetwork)
    targetSize = inputSizeOrNetwork;
else
    try
        targetSize = inputSizeOrNetwork.Layers(1).InputSize;
    catch
        error('DrishtiMitra:PreprocessInputSize', ...
            'Supply an input-size vector or a network whose first layer has InputSize.');
    end
end
if ~(isnumeric(targetSize) && numel(targetSize) >= 3 && all(targetSize(1:2) > 0) && targetSize(3) == 3)
    error('DrishtiMitra:PreprocessInputSize', ...
        'Classifier input size must specify positive height/width and exactly 3 channels.');
end
targetSize = double(targetSize(1:3));
end
