function metricsReport = evaluateAttentionUNet(net, datasetReport, imageSet, scaleFactor, threshold)
%EVALUATEATTENTIONUNET Evaluate trained model on test/val set ignoring unannotated channels.
%
%   metricsReport = evaluateAttentionUNet(net, datasetReport, imageSet, scaleFactor, threshold)
%   - net: trained dlnetwork
%   - datasetReport: output from validateIDRiDMasks() (optional)
%   - imageSet: "testing" (default, 27 images) or "training" (54 images)
%   - scaleFactor: default 0.5
%   - threshold: default 0.5

if nargin < 2 || isempty(datasetReport)
    datasetReport = validateIDRiDMasks();
end
if nargin < 3 || isempty(imageSet)
    imageSet = "testing";
end
if nargin < 4 || isempty(scaleFactor)
    scaleFactor = 0.5;
end
if nargin < 5 || isempty(threshold)
    threshold = 0.5;
end

images = datasetReport.images(datasetReport.images.ImageSet == imageSet, :);
nImages = height(images);
assert(nImages > 0, 'DrishtiMitra:EvalImages', 'No images found for set: %s', imageSet);

types = ["MA", "HE", "EX", "SE", "OD"];

perImageDice = cell(5, 1);
perImageIoU = cell(5, 1);
perImageSens = cell(5, 1);
perImageSpec = cell(5, 1);

for k = 1:5
    perImageDice{k} = [];
    perImageIoU{k} = [];
    perImageSens{k} = [];
    perImageSpec{k} = [];
end

fprintf('\nEvaluating Attention U-Net on %d %s set images...\n', nImages, imageSet);

for i = 1:nImages
    imgID = images.ID(i);
    imgPath = images.ImageFile(i);
    img = imread(imgPath);
    if ismatrix(img)
        img = repmat(img, 1, 1, 3);
    end
    
    imgScaled = imresize(img, scaleFactor, 'bilinear');
    [H, W, ~] = size(imgScaled);
    
    [~, binaryPred] = predictAttentionUNet(net, imgScaled, [512, 512], [256, 256], threshold);
    
    for k = 1:numel(types)
        typeMatches = false(height(datasetReport.records), 1);
        for r = 1:height(datasetReport.records)
            typeMatches(r) = matchesType(datasetReport.records.Type(r), types(k));
        end
        rows = datasetReport.records(datasetReport.records.ID == imgID & ...
            datasetReport.records.ImageSet == imageSet & typeMatches, :);
        
        if height(rows) == 1
            rawMask = readValidatedBinaryMask(rows.MaskFile(1));
            gtMask = imresize(rawMask, [H, W], 'nearest') > 0;
            predMask = binaryPred(:,:,k);
            
            TP = double(nnz(predMask & gtMask));
            FP = double(nnz(predMask & ~gtMask));
            FN = double(nnz(~predMask & gtMask));
            TN = double(nnz(~predMask & ~gtMask));
            
            if (TP + FP + FN) == 0
                dice = 1.0;
                iou = 1.0;
                sens = 1.0;
                spec = 1.0;
            else
                dice = (2 * TP) / (2 * TP + FP + FN + eps);
                iou = TP / (TP + FP + FN + eps);
                sens = TP / (TP + FN + eps);
                spec = TN / (TN + FP + eps);
            end
            
            perImageDice{k}(end+1, 1) = dice;
            perImageIoU{k}(end+1, 1) = iou;
            perImageSens{k}(end+1, 1) = sens;
            perImageSpec{k}(end+1, 1) = spec;
        end
    end
end

% Summarize metrics per channel
Channel = types';
EvaluatedImages = zeros(5, 1);
MeanDice = zeros(5, 1);
MeanIoU = zeros(5, 1);
MeanSensitivity = zeros(5, 1);
MeanSpecificity = zeros(5, 1);

for k = 1:5
    EvaluatedImages(k) = numel(perImageDice{k});
    if EvaluatedImages(k) > 0
        MeanDice(k) = mean(perImageDice{k});
        MeanIoU(k) = mean(perImageIoU{k});
        MeanSensitivity(k) = mean(perImageSens{k});
        MeanSpecificity(k) = mean(perImageSpec{k});
    end
end

summaryTable = table(Channel, EvaluatedImages, MeanDice, MeanIoU, MeanSensitivity, MeanSpecificity);

metricsReport = struct(...
    'imageSet', imageSet, ...
    'numImagesEvaluated', nImages, ...
    'summaryTable', summaryTable, ...
    'perImageDice', {perImageDice}, ...
    'perImageIoU', {perImageIoU}, ...
    'perImageSens', {perImageSens}, ...
    'perImageSpec', {perImageSpec});

fprintf('\n========================================================================================\n');
fprintf('  ATTENTION U-NET SEGMENTATION EVALUATION RESULTS (%s SET)\n', upper(imageSet));
fprintf('========================================================================================\n');
disp(summaryTable);
fprintf('========================================================================================\n\n');
end

function tf = matchesType(recordType, targetType)
rec = upper(string(recordType));
tgt = upper(string(targetType));
switch tgt
    case "MA"
        tf = contains(rec,"MA") || contains(rec,"MICROANEURYSM");
    case "HE"
        tf = contains(rec,"HE") || contains(rec,"HAEMORRHAGE");
    case "EX"
        tf = (contains(rec,"EX") && ~contains(rec,"SOFT")) || contains(rec,"HARD");
    case "SE"
        tf = contains(rec,"SE") || contains(rec,"SOFT");
    case "OD"
        tf = contains(rec,"OD") || contains(rec,"OPTIC");
    otherwise
        tf = (rec == tgt);
end
end

function mask = readValidatedBinaryMask(maskFile)
info = imfinfo(maskFile);
assert(numel(info) == 1, 'DrishtiMitra:IDRiDMaskPages', 'TIFF has %d pages: %s', numel(info), maskFile);
raw = imread(maskFile);
if ismatrix(raw)
    mask = raw ~= mode(raw(:));
    return
end
rgb = raw(:,:,1:3);
pixelRows = reshape(permute(rgb,[3 1 2]),3,[])';
[colours,~,colourIndex] = unique(pixelRows,'rows','stable');
counts = accumarray(colourIndex,1);
[~,backgroundIndex] = max(counts);
mask = reshape(colourIndex ~= backgroundIndex, size(raw,1), size(raw,2));
end
