% export_model.m
% Verify and export MATLAB-trained EfficientNet-B0 V2 classifier to ONNX format

addpath(genpath('d:\Projects\dmAICodex\DrishtiMitra_MATLAB'));

fprintf('=== PHASE 1: MATLAB ONNX EXPORT VERIFICATION ===\n\n');

mFile = 'd:/Projects/dmAICodex/DrishtiMitra_MATLAB/models/dr_classifier_efficientnetb0_V2.mat';
assert(isfile(mFile), 'Model file not found: %s', mFile);

fprintf('1. Loading MATLAB checkpoint: %s\n', mFile);
mStruct = load(mFile);

assert(isfield(mStruct, 'netV2'), 'Field netV2 not found in MAT file!');
net = mStruct.netV2;
info = mStruct.infoV2;

fprintf('   - Variable Name : netV2\n');
fprintf('   - Network Class : %s\n', class(net));
fprintf('   - Layer Count   : %d\n', numel(net.Layers));

inputLayer = net.Layers(1);
outputLayer = net.Layers(end);

fprintf('   - Input Layer   : %s (%s)\n', inputLayer.Name, class(inputLayer));
if isprop(inputLayer, 'InputSize')
    fprintf('   - Input Size    : %s\n', mat2str(inputLayer.InputSize));
end

fprintf('   - Output Layer  : %s (%s)\n', outputLayer.Name, class(outputLayer));
if isprop(outputLayer, 'Classes')
    fprintf('   - Classes       : %s\n', strjoin(string(outputLayer.Classes), ', '));
end

fprintf('\n2. Exporting to ONNX format...\n');

targetDir = 'd:/Projects/drishtiMitra/artifacts';
if ~isfolder(targetDir)
    mkdir(targetDir);
end
onnxPath = fullfile(targetDir, 'dr_classifier_v2.onnx');

try
    exportONNXNetwork(net, onnxPath);
    fprintf('   - Export Status : SUCCESS\n');
    fprintf('   - ONNX Path     : %s\n', onnxPath);
    
    fileInfo = dir(onnxPath);
    fprintf('   - ONNX Size     : %.2f MB (%d bytes)\n', fileInfo.bytes / (1024*1024), fileInfo.bytes);
    assert(fileInfo.bytes > 0, 'ONNX file is empty!');
catch ME
    fprintf('   - Export Status : FAILED\n');
    fprintf('   - Error Message : %s\n', ME.message);
    rethrow(ME);
end

fprintf('\n=== ONNX EXPORT COMPLETE ===\n');
