function results = trainAttentionUNet(opts)
%TRAINATTENTIONUNET Train Attention U-Net using 5-Fold CV or Final Full Training.
%
%   results = trainAttentionUNet(opts)
%   - opts: struct with fields:
%     .mode: "cv" (5-fold cross-validation) or "final" (full 54-image training, default)
%     .numEpochs: default 50
%     .batchSize: default 4
%     .learningRate: default 0.001
%     .patchesPerImage: default 8
%     .scaleFactor: default 0.5
%     .numFolds: default 5
%     .randomSeed: default 42
%     .saveDir: default fullfile(projectRoot, "models")
%     .evaluateTestSet: default true (only for "final" mode, on 27 test images)

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

if nargin < 1 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'mode') || isempty(opts.mode), opts.mode = "final"; end
if ~isfield(opts, 'numEpochs') || isempty(opts.numEpochs), opts.numEpochs = 50; end
if ~isfield(opts, 'batchSize') || isempty(opts.batchSize), opts.batchSize = 4; end
if ~isfield(opts, 'learningRate') || isempty(opts.learningRate), opts.learningRate = 0.001; end
if ~isfield(opts, 'patchesPerImage') || isempty(opts.patchesPerImage), opts.patchesPerImage = 8; end
if ~isfield(opts, 'scaleFactor') || isempty(opts.scaleFactor), opts.scaleFactor = 0.5; end
if ~isfield(opts, 'numFolds') || isempty(opts.numFolds), opts.numFolds = 5; end
if ~isfield(opts, 'randomSeed') || isempty(opts.randomSeed), opts.randomSeed = 42; end
if ~isfield(opts, 'saveDir') || isempty(opts.saveDir), opts.saveDir = fullfile(projectRoot, "models"); end
if ~isfield(opts, 'evaluateTestSet') || isempty(opts.evaluateTestSet), opts.evaluateTestSet = true; end

if ~isfolder(opts.saveDir)
    mkdir(opts.saveDir);
end

fprintf('\n========================================================================================\n');
fprintf('  DRISHTIMITRA: ATTENTION U-NET TRAINING WORKFLOW (MODE: %s)\n', upper(opts.mode));
fprintf('========================================================================================\n');
fprintf('Configuration:\n');
fprintf('  Epochs: %d | Batch Size: %d | Learning Rate: %g\n', opts.numEpochs, opts.batchSize, opts.learningRate);
fprintf('  Patches/Image: %d | Scale Factor: %.2f | Seed: %d\n', opts.patchesPerImage, opts.scaleFactor, opts.randomSeed);
fprintf('  Save Directory: %s\n', opts.saveDir);
fprintf('========================================================================================\n\n');

datasetReport = validateIDRiDMasks();
trainImages = datasetReport.images(datasetReport.images.ImageSet == "training", :);
nTrain = height(trainImages);
assert(nTrain == 54, 'DrishtiMitra:TrainImages', 'Expected exactly 54 training images, found %d.', nTrain);

rng(opts.randomSeed);

if opts.mode == "cv"
    results = runCrossValidation(trainImages, datasetReport, opts);
elseif opts.mode == "final"
    results = runFinalTraining(trainImages, datasetReport, opts);
else
    error('DrishtiMitra:TrainMode', 'Unsupported mode: %s. Use "cv" or "final".', opts.mode);
end
end

function cvResults = runCrossValidation(trainImages, datasetReport, opts)
nTrain = height(trainImages);
numFolds = opts.numFolds;

if numFolds >= 2
    foldIndices = cvpartition(nTrain, 'KFold', numFolds);
else
    holdoutPart = cvpartition(nTrain, 'HoldOut', 0.2);
end

foldModels = cell(numFolds, 1);
foldHistories = cell(numFolds, 1);
foldValReports = cell(numFolds, 1);

fprintf('Starting %d-Fold Cross-Validation on 54 official training images...\n\n', numFolds);

for fold = 1:numFolds
    fprintf('>>> FOLD %d / %d <<<\n', fold, numFolds);
    if numFolds >= 2
        trainIdx = training(foldIndices, fold);
        valIdx = test(foldIndices, fold);
    else
        trainIdx = training(holdoutPart);
        valIdx = test(holdoutPart);
    end
    
    trainFoldImgs = trainImages(trainIdx, :);
    valFoldImgs = trainImages(valIdx, :);
    
    % Build fresh model for fold
    net = buildAttentionUNet([512, 512, 3], 5, [16, 32, 64, 128, 256]);
    
    % Train fold model (dynamic per-epoch channel-aware patch resampling)
    [net, history] = trainModelLoop(net, trainFoldImgs, datasetReport, opts, sprintf('Fold %d', fold), fold);
    
    % Evaluate fold model on validation fold images
    fprintf('Evaluating Fold %d model on validation fold images...\n', fold);
    valReport = evaluateSubset(net, valFoldImgs, datasetReport, opts.scaleFactor, 0.5);
    
    foldModels{fold} = net;
    foldHistories{fold} = history;
    foldValReports{fold} = valReport;
    
    % Save fold checkpoint
    foldFile = fullfile(opts.saveDir, sprintf('attention_unet_cv_fold%d.mat', fold));
    save(foldFile, 'net', 'history', 'valReport', 'opts', 'trainFoldImgs', 'valFoldImgs');
    fprintf('Saved fold %d checkpoint to %s\n\n', fold, foldFile);
end

cvResults = struct(...
    'mode', "cv", ...
    'numFolds', numFolds, ...
    'foldModels', {foldModels}, ...
    'foldHistories', {foldHistories}, ...
    'foldValReports', {foldValReports}, ...
    'opts', opts);
end

function finalResults = runFinalTraining(trainImages, datasetReport, opts)
fprintf('Starting FINAL training on all 54 official training images...\n\n');

% Build fresh model
net = buildAttentionUNet([512, 512, 3], 5, [16, 32, 64, 128, 256]);

% Train model (dynamic per-epoch channel-aware patch resampling)
[net, history] = trainModelLoop(net, trainImages, datasetReport, opts, 'Final Model', 0);

% Save final model checkpoint
finalModelFile = fullfile(opts.saveDir, 'attention_unet_final.mat');
timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
save(finalModelFile, 'net', 'history', 'opts', 'timestamp');
fprintf('Saved final trained model checkpoint to: %s\n\n', finalModelFile);

testReport = [];
if opts.evaluateTestSet
    fprintf('Evaluating final model on 27 official held-out testing images...\n');
    testReport = evaluateAttentionUNet(net, datasetReport, "testing", opts.scaleFactor, 0.5);
    reportFile = fullfile(opts.saveDir, 'attention_unet_final_test_report.mat');
    save(reportFile, 'testReport', 'opts', 'timestamp');
    fprintf('Saved final test evaluation report to: %s\n', reportFile);
end

finalResults = struct(...
    'mode', "final", ...
    'net', net, ...
    'history', history, ...
    'testReport', testReport, ...
    'checkpointFile', string(finalModelFile), ...
    'opts', opts);
end

function [net, history] = trainModelLoop(net, trainSubImages, datasetReport, opts, label, foldSeedOffset)
batchSize = opts.batchSize;
hasGPU = canUseGPU();
learningRate = opts.learningRate;
averageGrad = [];
averageSqGrad = [];

epochLosses = zeros(opts.numEpochs, 1);
epochBCE = zeros(opts.numEpochs, 1);
epochDice = zeros(opts.numEpochs, 1);

tTrain = tic;

for epoch = 1:opts.numEpochs
    % Re-generate fresh channel-aware patches dynamically at start of each epoch
    epochSeed = opts.randomSeed + foldSeedOffset * 1000 + epoch;
    patches = extractIDRiDPatches(datasetReport, trainSubImages, [512, 512], opts.patchesPerImage, opts.scaleFactor, epochSeed);
    
    nPatches = patches.totalPatches;
    nBatches = floor(nPatches / batchSize);
    perm = randperm(nPatches);
    
    batchLossSum = 0;
    bceSum = 0;
    diceSum = 0;
    
    for b = 1:nBatches
        batchIdx = perm((b-1)*batchSize + 1 : b*batchSize);
        
        X_raw = zeros(512, 512, 3, batchSize, 'single');
        Y_raw = zeros(512, 512, 5, batchSize, 'single');
        V_raw = zeros(1, 1, 5, batchSize, 'single');
        
        for k = 1:batchSize
            idx = batchIdx(k);
            X_raw(:,:,:,k) = single(patches.images{idx}) / 255.0;
            Y_raw(:,:,:,k) = single(patches.masks{idx});
            V_raw(1,1,:,k) = single(patches.validity{idx});
        end
        
        X = dlarray(X_raw, 'SSCB');
        Y = dlarray(Y_raw, 'SSCB');
        V = dlarray(V_raw, 'SSCB');
        
        if hasGPU
            X = gpuArray(X);
            Y = gpuArray(Y);
            V = gpuArray(V);
        end
        
        [loss, gradients, state, lossMetrics] = dlfeval(@computeMaskedMultilabelLoss, net, X, Y, V);
        net.State = state;
        
        % Global iteration step count for Adam schedule
        iterStep = (epoch - 1) * nBatches + b;
        
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iterStep, learningRate);
        
        batchLossSum = batchLossSum + double(extractdata(loss));
        bceSum = bceSum + double(lossMetrics.bceLoss);
        diceSum = diceSum + double(lossMetrics.diceLoss);
    end
    
    epochLosses(epoch) = batchLossSum / nBatches;
    epochBCE(epoch) = bceSum / nBatches;
    epochDice(epoch) = diceSum / nBatches;
    
    if mod(epoch, 5) == 0 || epoch == 1 || epoch == opts.numEpochs
        fprintf('[%s] Epoch %3d/%3d — Loss: %.4f (BCE: %.4f, Dice: %.4f)\n', ...
            label, epoch, opts.numEpochs, epochLosses(epoch), epochBCE(epoch), epochDice(epoch));
    end
end

elapsedTime = toc(tTrain);
fprintf('[%s] Training completed in %.2f sec (%.2f min).\n', label, elapsedTime, elapsedTime/60);

history = struct(...
    'epochLosses', epochLosses, ...
    'epochBCE', epochBCE, ...
    'epochDice', epochDice, ...
    'elapsedTimeSec', elapsedTime);
end

function patches = extractPatchesFromSubset(subImages, datasetReport, patchesPerImage, scaleFactor, seed)
patches = extractIDRiDPatches(datasetReport, subImages, [512, 512], patchesPerImage, scaleFactor, seed);
end


function report = evaluateSubset(net, subImages, datasetReport, scaleFactor, threshold)
nImages = height(subImages);
types = ["MA", "HE", "EX", "SE", "OD"];
perImageDice = cell(5, 1);
for k = 1:5, perImageDice{k} = []; end

for i = 1:nImages
    imgID = subImages.ID(i);
    imgPath = subImages.ImageFile(i);
    imgSet = subImages.ImageSet(i);
    img = imread(imgPath);
    if ismatrix(img), img = repmat(img, 1, 1, 3); end
    imgScaled = imresize(img, scaleFactor, 'bilinear');
    [H, W, ~] = size(imgScaled);
    
    [~, binaryPred] = predictAttentionUNet(net, imgScaled, [512, 512], [256, 256], threshold);
    
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
            predMask = binaryPred(:,:,k);
            
            TP = double(nnz(predMask & gtMask));
            FP = double(nnz(predMask & ~gtMask));
            FN = double(nnz(~predMask & gtMask));
            
            if (TP + FP + FN) == 0
                dice = 1.0;
            else
                dice = (2 * TP) / (2 * TP + FP + FN + eps);
            end
            perImageDice{k}(end+1, 1) = dice;
        end
    end
end

MeanDice = zeros(5, 1);
for k = 1:5
    if ~isempty(perImageDice{k})
        MeanDice(k) = mean(perImageDice{k});
    end
end

report = table(types', MeanDice, 'VariableNames', {'Channel', 'MeanDice'});
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
