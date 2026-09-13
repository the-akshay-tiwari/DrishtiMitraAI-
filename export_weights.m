% export_weights.m
% Extract weights from MATLAB-trained EfficientNet-B0 V2 (DAGNetwork)

addpath(genpath('d:\Projects\dmAICodex\DrishtiMitra_MATLAB'));

fprintf('=== EXPORTING MATLAB NETV2 WEIGHTS ===\n\n');

mFile = 'd:/Projects/dmAICodex/DrishtiMitra_MATLAB/models/dr_classifier_efficientnetb0_V2.mat';
mStruct = load(mFile);
net = mStruct.netV2;
info = mStruct.infoV2;

% Find head layer 'dr_head'
layers = net.Layers;
fcIdx = find(arrayfun(@(x) strcmp(x.Name, 'dr_head'), layers), 1);
if isempty(fcIdx)
    fcIdx = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.FullyConnectedLayer'), layers), 1, 'last');
end

headWeights = layers(fcIdx).Weights; % size [5, 1280]
headBias = layers(fcIdx).Bias;       % size [5, 1]

fprintf('Head Layer Name    : %s\n', layers(fcIdx).Name);
fprintf('Head Weights Size  : %s\n', mat2str(size(headWeights)));
fprintf('Head Bias Size     : %s\n', mat2str(size(headBias)));

% Target output file
targetDir = 'd:/Projects/drishtiMitra/artifacts';
if ~isfolder(targetDir)
    mkdir(targetDir);
end
outFile = fullfile(targetDir, 'matlab_dr_classifier_v2_weights.mat');

save(outFile, 'headWeights', 'headBias', 'info', '-v7');
fprintf('Saved MATLAB classifier weights to: %s\n', outFile);
fprintf('=== WEIGHT EXPORT COMPLETE ===\n');
