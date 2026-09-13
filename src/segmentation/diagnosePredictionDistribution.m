function report = diagnosePredictionDistribution(checkpointFile, numValImages)
%DIAGNOSEPREDICTIONDISTRIBUTION Inspect probMap distributions, threshold sensitivity, and GT alignment.
%
%   report = diagnosePredictionDistribution(checkpointFile, numValImages)
%   - checkpointFile: default 'models/attention_unet_cv_fold1.mat'
%   - numValImages: default 3

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

if nargin < 1 || isempty(checkpointFile)
    checkpointFile = fullfile(projectRoot, 'models', 'attention_unet_cv_fold1.mat');
end
if nargin < 2 || isempty(numValImages)
    numValImages = 3;
end

assert(isfile(checkpointFile), 'DrishtiMitra:DiagFile', ...
    'Checkpoint file not found: %s', checkpointFile);

fprintf('\nLoading checkpoint: %s...\n', checkpointFile);
ckpt = load(checkpointFile);
net = ckpt.net;
valFoldImgs = ckpt.valFoldImgs;
datasetReport = validateIDRiDMasks();

nVal = min(numValImages, height(valFoldImgs));
types = ["MA", "HE", "EX", "SE", "OD"];

% Pre-allocate diagnostic records
recImgID = [];
recChannel = [];
recMinProb = [];
recMaxProb = [];
recMeanProb = [];
recP95Prob = [];
recGTCount = [];
recPosCount_01 = [];
recPosCount_025 = [];
recPosCount_05 = [];
recDice_01 = [];
recDice_025 = [];
recDice_05 = [];

scaleFactor = 0.5;

for i = 1:nVal
    imgID = valFoldImgs.ID(i);
    imgPath = valFoldImgs.ImageFile(i);
    imgSet = valFoldImgs.ImageSet(i);
    
    img = imread(imgPath);
    if ismatrix(img), img = repmat(img, 1, 1, 3); end
    imgScaled = imresize(img, scaleFactor, 'bilinear');
    [H, W, ~] = size(imgScaled);
    
    probMap = predictAttentionUNet(net, imgScaled, [512, 512], [256, 256]);
    
    for k = 1:5
        typeMatches = false(height(datasetReport.records), 1);
        for r = 1:height(datasetReport.records)
            typeMatches(r) = matchesType(datasetReport.records.Type(r), types(k));
        end
        rows = datasetReport.records(datasetReport.records.ID == imgID & ...
            datasetReport.records.ImageSet == imgSet & typeMatches, :);
        
        if height(rows) == 1
            rawMask = readValidatedBinaryMask(rows.MaskFile(1));
            gtMask = imresize(rawMask, [H, W], 'nearest') > 0;
            pChannel = probMap(:,:,k);
            
            pVec = double(pChannel(:));
            minP = min(pVec);
            maxP = max(pVec);
            meanP = mean(pVec);
            p95 = prctile(pVec, 95);
            
            gtCount = nnz(gtMask);
            
            bin01 = pChannel >= 0.10;
            bin025 = pChannel >= 0.25;
            bin05 = pChannel >= 0.50;
            
            cnt01 = nnz(bin01);
            cnt025 = nnz(bin025);
            cnt05 = nnz(bin05);
            
            d01 = computeDice(bin01, gtMask);
            d025 = computeDice(bin025, gtMask);
            d05 = computeDice(bin05, gtMask);
            
            recImgID = [recImgID; string(imgID)]; %#ok<AGROW>
            recChannel = [recChannel; types(k)]; %#ok<AGROW>
            recMinProb = [recMinProb; minP]; %#ok<AGROW>
            recMaxProb = [recMaxProb; maxP]; %#ok<AGROW>
            recMeanProb = [recMeanProb; meanP]; %#ok<AGROW>
            recP95Prob = [recP95Prob; p95]; %#ok<AGROW>
            recGTCount = [recGTCount; gtCount]; %#ok<AGROW>
            recPosCount_01 = [recPosCount_01; cnt01]; %#ok<AGROW>
            recPosCount_025 = [recPosCount_025; cnt025]; %#ok<AGROW>
            recPosCount_05 = [recPosCount_05; cnt05]; %#ok<AGROW>
            recDice_01 = [recDice_01; d01]; %#ok<AGROW>
            recDice_025 = [recDice_025; d025]; %#ok<AGROW>
            recDice_05 = [recDice_05; d05]; %#ok<AGROW>
        end
    end
end

summaryTable = table(recImgID, recChannel, recMinProb, recMaxProb, recMeanProb, recP95Prob, ...
    recGTCount, recPosCount_01, recPosCount_025, recPosCount_05, ...
    recDice_01, recDice_025, recDice_05, ...
    'VariableNames', {'ImageID', 'Channel', 'MinP', 'MaxP', 'MeanP', 'P95', 'GTPixels', 'PosP_01', 'PosP_025', 'PosP_05', 'Dice_01', 'Dice_025', 'Dice_05'});

fprintf('\n====================================================================================================\n');
fprintf('  PROBABILITY MAP & THRESHOLD SENSITIVITY DIAGNOSTIC REPORT (%d Validation Images)\n', nVal);
fprintf('====================================================================================================\n');
disp(summaryTable);
fprintf('====================================================================================================\n\n');

% Render diagnostic visual figure for Image 1
fig = renderDiagnosticVisual(net, valFoldImgs.ID(1), valFoldImgs.ImageFile(1), valFoldImgs.ImageSet(1), datasetReport, scaleFactor);

outDir = fullfile(projectRoot, 'artifacts_output');
if ~isfolder(outDir), mkdir(outDir); end
outFile = fullfile(outDir, 'diagnose_pred_distribution.png');
exportgraphics(fig, outFile);
fprintf('Saved diagnostic heatmap visual figure to: %s\n', outFile);

report = struct(...
    'summaryTable', summaryTable, ...
    'figureFile', string(outFile));
end

function d = computeDice(predMask, gtMask)
TP = double(nnz(predMask & gtMask));
FP = double(nnz(predMask & ~gtMask));
FN = double(nnz(~predMask & gtMask));
if (TP + FP + FN) == 0
    d = 1.0;
else
    d = (2 * TP) / (2 * TP + FP + FN + eps);
end
end

function fig = renderDiagnosticVisual(net, imgID, imgPath, imgSet, datasetReport, scaleFactor)
img = imread(imgPath);
if ismatrix(img), img = repmat(img, 1, 1, 3); end
imgScaled = imresize(img, scaleFactor, 'bilinear');
[H, W, ~] = size(imgScaled);

probMap = predictAttentionUNet(net, imgScaled, [512, 512], [256, 256]);

% Channels to display: MA (1), EX (3), OD (5)
chIdx = [1, 3, 5];
chNames = ["MA", "EX", "OD"];

fig = figure('Name', sprintf('Diagnostic Heatmaps: %s', imgID), ...
    'Color', 'w', 'NumberTitle', 'off', 'Position', [100 100 1500 900]);
layout = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
title(layout, sprintf('Diagnostic Probability Heatmaps & GT Alignment — %s', imgID), 'FontWeight', 'bold');

for i = 1:3
    k = chIdx(i);
    cName = chNames(i);
    
    % GT Mask
    typeMatches = false(height(datasetReport.records), 1);
    for r = 1:height(datasetReport.records)
        typeMatches(r) = matchesType(datasetReport.records.Type(r), cName);
    end
    rows = datasetReport.records(datasetReport.records.ID == imgID & ...
        datasetReport.records.ImageSet == imgSet & typeMatches, :);
    
    if height(rows) == 1
        rawMask = readValidatedBinaryMask(rows.MaskFile(1));
        gtMask = imresize(rawMask, [H, W], 'nearest') > 0;
    else
        gtMask = false(H, W);
    end
    
    % 1. Input image
    nexttile; imshow(imgScaled); title(sprintf('%s: Input Image', cName));
    
    % 2. Ground truth mask
    nexttile; imshow(gtMask); title(sprintf('%s: Ground Truth Mask', cName));
    
    % 3. Probability Heatmap
    nexttile; imshow(probMap(:,:,k), []); colormap(gca, 'jet'); colorbar;
    title(sprintf('%s: Prob Heatmap (Max: %.3f)', cName, max(probMap(:,:,k), [], 'all')));
    
    % 4. Pred @ threshold 0.1 vs 0.5
    nexttile;
    comp = zeros(H, W, 3);
    comp(:,:,1) = double(probMap(:,:,k) >= 0.10); % Red = Pred @ 0.10
    comp(:,:,2) = double(gtMask);                % Green = GT
    imshow(comp);
    title(sprintf('%s: Red=Pred@0.10, Green=GT', cName));
end
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
