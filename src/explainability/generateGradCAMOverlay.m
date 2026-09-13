function overlay = generateGradCAMOverlay(net, I, classIndex)
%GENERATEGRADCAMOVERLAY Generate Grad-CAM overlay using final EfficientNet layer.
%
%   OVERLAY = generateGradCAMOverlay(NET, I, CLASSINDEX)
%
%   CLASSINDEX:
%       1 = Grade 0
%       2 = Grade 1
%       3 = Grade 2
%       4 = Grade 3
%       5 = Grade 4

arguments
    net
    I {mustBeNumeric}
    classIndex (1,1) double {mustBeInteger,mustBePositive}
end

%% Validate class index

if classIndex < 1 || classIndex > 5
    error('DrishtiMitra:GradCAMClassIndex', ...
        'Class index must be between 1 and 5.');
end

%% Prepare image

if ndims(I) == 2
    I = cat(3,I,I,I);
elseif ndims(I) == 3 && size(I,3) == 1
    I = cat(3,I,I,I);
elseif ndims(I) ~= 3 || size(I,3) ~= 3
    error('DrishtiMitra:GradCAMImage', ...
        'Input image must be M-by-N grayscale or M-by-N-by-3 RGB.');
end

%% Convert image to uint8

if isa(I,'uint8')

    Idisplay = I;

elseif isa(I,'single') || isa(I,'double')

    if any(~isfinite(I),'all')
        error('DrishtiMitra:GradCAMImageValues', ...
            'Input image contains non-finite values.');
    end

    minValue = min(I,[],'all');
    maxValue = max(I,[],'all');

    if minValue >= 0 && maxValue <= 1
        Idisplay = im2uint8(I);

    elseif minValue >= 0 && maxValue <= 255
        Idisplay = uint8(round(I));

    else
        error('DrishtiMitra:GradCAMImageRange', ...
            'Floating-point image must be in [0,1] or [0,255].');
    end

else
    Idisplay = im2uint8(I);
end

%% EfficientNet-B0 final convolution layer

featureLayer = ...
    'efficientnet-b0|model|head|conv2d|Conv2D';

%% Generate Grad-CAM

try

    scoreMap = gradCAM( ...
        net, ...
        Idisplay, ...
        classIndex, ...
        'FeatureLayer', featureLayer);

catch ME

    error('DrishtiMitra:GradCAMFailed', ...
        ['Grad-CAM generation failed.\n\n' ...
         'Feature layer: %s\n' ...
         'Class index: %d\n\n' ...
         'Original error:\n%s'], ...
         featureLayer, classIndex, ME.message);

end

%% Normalize Grad-CAM map

scoreMap = double(scoreMap);

if any(~isfinite(scoreMap),'all')
    error('DrishtiMitra:GradCAMMap', ...
        'Grad-CAM returned non-finite values.');
end

scoreMap = rescale(scoreMap,0,1);

%% Resize heatmap to original image

scoreMap = imresize( ...
    scoreMap, ...
    size(Idisplay,[1 2]), ...
    'bilinear');

scoreMap = min(max(scoreMap,0),1);

%% Convert score map to heatmap

heatmapRGB = ind2rgb( ...
    im2uint8(scoreMap), ...
    hot(256));

%% Blend with original image

overlay = imfuse( ...
    Idisplay, ...
    heatmapRGB, ...
    'blend');

%% Ensure uint8 output

if ~isa(overlay,'uint8')
    overlay = im2uint8(overlay);
end

end