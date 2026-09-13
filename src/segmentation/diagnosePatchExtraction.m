function report = diagnosePatchExtraction(numPatchesPerImage, randomSeed)
%DIAGNOSEPATCHEXTRACTION Inspect extracted training patches for positive pixel counts and lesion coverage.
%
%   report = diagnosePatchExtraction(numPatchesPerImage, randomSeed)
%   - numPatchesPerImage: default 8 (total 432 patches for 54 training images)
%   - randomSeed: default 42

if nargin < 1 || isempty(numPatchesPerImage)
    numPatchesPerImage = 8;
end
if nargin < 2 || isempty(randomSeed)
    randomSeed = 42;
end

datasetReport = validateIDRiDMasks();
patches = extractIDRiDPatches(datasetReport, "training", [512, 512], numPatchesPerImage, 0.5, randomSeed);

nPatches = patches.totalPatches;
types = ["MA", "HE", "EX", "SE", "OD"];

positivePatchCounts = zeros(5, 1);
totalPositivePixels = zeros(5, 1);
validPatchCounts = zeros(5, 1);

for i = 1:nPatches
    msk = patches.masks{i}; % [512, 512, 5]
    val = patches.validity{i}; % [1, 5]
    
    for k = 1:5
        if val(k)
            validPatchCounts(k) = validPatchCounts(k) + 1;
            nPos = nnz(msk(:,:,k));
            if nPos > 0
                positivePatchCounts(k) = positivePatchCounts(k) + 1;
                totalPositivePixels(k) = totalPositivePixels(k) + nPos;
            end
        end
    end
end

PercentPositivePatches = (positivePatchCounts ./ max(1, validPatchCounts)) * 100;
MeanPositivePixelsPerPatch = totalPositivePixels ./ max(1, validPatchCounts);
TotalPixelsPerChannel = validPatchCounts * (512 * 512);
PositivePixelPercentage = (totalPositivePixels ./ max(1, TotalPixelsPerChannel)) * 100;

summaryTable = table(types', validPatchCounts, positivePatchCounts, PercentPositivePatches, ...
    MeanPositivePixelsPerPatch, PositivePixelPercentage, ...
    'VariableNames', {'LesionChannel', 'ValidPatches', 'PositivePatches', 'PercentPositivePatches', 'MeanPosPixelsPerPatch', 'PosPixelPercent'});

report = struct(...
    'totalPatchesExtracted', nPatches, ...
    'summaryTable', summaryTable, ...
    'patches', patches);

fprintf('\n========================================================================================\n');
fprintf('  PATCH EXTRACTION DIAGNOSTIC REPORT (%d Patches from 54 Training Images)\n', nPatches);
fprintf('========================================================================================\n');
disp(summaryTable);
fprintf('========================================================================================\n\n');
end
