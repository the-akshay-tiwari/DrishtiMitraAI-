function stats = diagnosePatchSamplingPipeline()
%DIAGNOSEPATCHSAMPLINGPIPELINE Diagnostic for new channel-aware patch sampling.

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
datasetReport = validateIDRiDMasks();

% Get the exact 43 training images used in Fold 1
trainImages = datasetReport.images(datasetReport.images.ImageSet == "training", :);
rng(42);
holdoutPart = cvpartition(height(trainImages), 'HoldOut', 0.2);
trainFoldImgs = trainImages(training(holdoutPart), :);

nTrainFold = height(trainFoldImgs);
fprintf('Fold 1 Training Image Count: %d images (Validation Fold Count: %d images)\n', ...
    nTrainFold, height(trainImages) - nTrainFold);

% Extract 1 epoch of patches (8 patches per image = 344 total patches)
patches = extractIDRiDPatches(datasetReport, trainFoldImgs, [512, 512], 8, 0.5, 42);

types = ["MA", "HE", "EX", "SE", "OD"];

fprintf('\n========================================================================================\n');
fprintf('  DRISHTIMITRA: CHANNEL-AWARE PATCH SAMPLING DIAGNOSTIC REPORT (43 Train Images)\n');
fprintf('========================================================================================\n');
fprintf('Total Extracted Patches : %d\n', patches.totalPatches);
fprintf('Pure Background Patches : %d (%.2f%%)\n', patches.pureBackgroundCount, (patches.pureBackgroundCount/patches.totalPatches)*100);
fprintf('Multi-Channel Pos Patches: %d (%.2f%%)\n', patches.multiPosCount, (patches.multiPosCount/patches.totalPatches)*100);
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('Per-Channel Positive Patch Statistics:\n');

summaryT = table(types', patches.channelPosCounts', patches.channelPosPct', ...
    'VariableNames', {'Channel', 'PositivePatchCount', 'PositivePatchPercentage'});
disp(summaryT);
fprintf('========================================================================================\n\n');

% Verify epoch 1 vs epoch 2 seed difference
patchesE1 = extractIDRiDPatches(datasetReport, trainFoldImgs, [512, 512], 8, 0.5, 43);
patchesE2 = extractIDRiDPatches(datasetReport, trainFoldImgs, [512, 512], 8, 0.5, 44);

diffImg1 = patchesE1.images{1};
diffImg2 = patchesE2.images{1};
isNewPatchesEachEpoch = ~isequal(diffImg1, diffImg2);

fprintf('Verification: Dynamic Per-Epoch Resampling (Epoch 1 vs Epoch 2): %s\n', ...
    string(isNewPatchesEachEpoch));

stats = struct(...
    'totalPatches', patches.totalPatches, ...
    'channelPosCounts', patches.channelPosCounts, ...
    'channelPosPct', patches.channelPosPct, ...
    'multiPosCount', patches.multiPosCount, ...
    'pureBackgroundCount', patches.pureBackgroundCount, ...
    'isNewPatchesEachEpoch', isNewPatchesEachEpoch);
end
