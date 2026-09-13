function fig = visualizeAttentionUNetPredictions(net, imageID, datasetReport, outputDir)
%VISUALIZEATTENTIONUNETPREDICTIONS Render predictions vs GT masks for one IDRiD image.
%
%   fig = visualizeAttentionUNetPredictions(net, imageID, datasetReport, outputDir)
%   - net: trained dlnetwork
%   - imageID: e.g. "IDRiD_055" or "55"
%   - datasetReport: output from validateIDRiDMasks() (optional)
%   - outputDir: optional directory path to save figure as PNG

if nargin < 3 || isempty(datasetReport)
    datasetReport = validateIDRiDMasks();
end
if nargin < 4
    outputDir = "";
end

id = normaliseID(imageID);
imgRow = datasetReport.images(datasetReport.images.ID == id, :);
assert(height(imgRow) == 1, 'DrishtiMitra:VisID', 'Image %s not found in dataset.', id);

imgFile = imgRow.ImageFile(1);
imgSet = imgRow.ImageSet(1);

origImg = imread(imgFile);
if ismatrix(origImg)
    origImg = repmat(origImg, 1, 1, 3);
end

scaleFactor = 0.5;
imgScaled = imresize(origImg, scaleFactor, 'bilinear');
[H, W, ~] = size(imgScaled);

[probMap, binaryPred] = predictAttentionUNet(net, imgScaled, [512, 512], [256, 256], 0.5);

types = ["MA", "HE", "EX", "SE", "OD"];
colours = [0.00 0.85 1.00; 1.00 0.10 0.10; 1.00 0.90 0.00; 1.00 0.10 0.85; 0.10 1.00 0.20];

gtMasks = cell(5, 1);
gtPresent = false(5, 1);

for k = 1:5
    typeMatches = false(height(datasetReport.records), 1);
    for r = 1:height(datasetReport.records)
        typeMatches(r) = matchesType(datasetReport.records.Type(r), types(k));
    end
    rows = datasetReport.records(datasetReport.records.ID == id & ...
        datasetReport.records.ImageSet == imgSet & typeMatches, :);
    if height(rows) == 1
        rawMask = readValidatedBinaryMask(rows.MaskFile(1));
        gtMasks{k} = imresize(rawMask, [H, W], 'nearest') > 0;
        gtPresent(k) = true;
    end
end

fig = figure('Name', sprintf('Predictions: %s (%s)', id, imgSet), ...
    'Color', 'w', 'NumberTitle', 'off', 'Position', [100 100 1400 800]);
layout = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
title(layout, sprintf('Attention U-Net Predictions vs GT — %s (%s set)', id, imgSet), 'FontWeight', 'bold');

% Tile 1: Original image
nexttile; imshow(imgScaled); title('Original Fundus');

% Tiles 2-6: Per-channel predictions vs GT
for k = 1:5
    nexttile;
    if gtPresent(k)
        % Composite GT (green) and Prediction (red) -> Overlap (yellow)
        comp = zeros(H, W, 3);
        comp(:,:,1) = double(binaryPred(:,:,k)); % Red = Pred
        comp(:,:,2) = double(gtMasks{k});        % Green = GT
        imshow(comp);
        title(sprintf('%s (Green: GT, Red: Pred)', types(k)));
    else
        imshow(binaryPred(:,:,k));
        title(sprintf('%s (Pred only - Not annotated)', types(k)));
    end
end

% Tile 7: Overlay of all predicted channels
nexttile;
imshow(imgScaled); hold on;
for k = 1:5
    colourLayer = uint8(255 * reshape(colours(k,:), 1, 1, 3));
    colourLayer = repmat(colourLayer, H, W);
    h = image(colourLayer);
    set(h, 'AlphaData', 0.4 * double(binaryPred(:,:,k)));
end
hold off; title('Pred Overlay (Cyan:MA, Red:HE, Yel:EX, Mag:SE, Grn:OD)');

% Tile 8: Summary Info
nexttile; axis off;
statusText = sprintf('Image: %s (%s set)\nResolution: %dx%d\n', id, imgSet, H, W);
for k = 1:5
    if gtPresent(k)
        dice = (2 * nnz(binaryPred(:,:,k) & gtMasks{k})) / ...
            max(1, (2*nnz(binaryPred(:,:,k) & gtMasks{k}) + nnz(binaryPred(:,:,k) & ~gtMasks{k}) + nnz(~binaryPred(:,:,k) & gtMasks{k})));
        statusText = [statusText, sprintf('%s Dice: %.4f\n', types(k), dice)]; %#ok<AGROW>
    else
        statusText = [statusText, sprintf('%s: GT Not Annotated\n', types(k))]; %#ok<AGROW>
    end
end
text(0.1, 0.5, statusText, 'FontSize', 10, 'FontWeight', 'bold');

if strlength(outputDir) > 0
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end
    outFile = fullfile(outputDir, sprintf('%s_attention_unet_pred.png', lower(id)));
    exportgraphics(fig, outFile);
    fprintf('Exported prediction visualization to %s\n', outFile);
end
end

function id = normaliseID(imageID)
token = regexp(char(imageID),'(?i)^(?:IDRiD_)?(\d{1,3})(?:test)?$','tokens','once');
assert(~isempty(token),'DrishtiMitra:VisID', 'imageID must have form IDRiD_# or #');
number = str2double(token{1});
id = string(sprintf('IDRiD_%03d',number));
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
