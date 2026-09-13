function net = buildAttentionUNet(inputSize, numClasses, filterList)
%BUILDATTENTIONUNET Create 4-stage Attention-Gated U-Net dlnetwork.
%
%   net = buildAttentionUNet(inputSize, numClasses, filterList)
%   - inputSize: [H, W, C], default [512, 512, 3]
%   - numClasses: number of independent sigmoid outputs, default 5
%   - filterList: filter counts per encoder level, default [16, 32, 64, 128, 256]

if nargin < 1 || isempty(inputSize)
    inputSize = [512, 512, 3];
end
if nargin < 2 || isempty(numClasses)
    numClasses = 5;
end
if nargin < 3 || isempty(filterList)
    filterList = [16, 32, 64, 128, 256];
end

assert(numel(filterList) == 5, 'DrishtiMitra:AttentionUNetFilters', ...
    'filterList must contain exactly 5 elements for a 4-stage U-Net.');

f1 = filterList(1); % 16
f2 = filterList(2); % 32
f3 = filterList(3); % 64
f4 = filterList(4); % 128
f5 = filterList(5); % 256

lgraph = layerGraph();

% --- Input ---
lgraph = addLayers(lgraph, imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input'));

% --- Encoder Level 1 (512x512) ---
enc1_layers = [
    convolution2dLayer(3, f1, 'Padding', 'same', 'Name', 'conv_e1_1')
    batchNormalizationLayer('Name', 'bn_e1_1')
    reluLayer('Name', 'relu_e1_1')
    convolution2dLayer(3, f1, 'Padding', 'same', 'Name', 'conv_e1_2')
    batchNormalizationLayer('Name', 'bn_e1_2')
    reluLayer('Name', 'relu_e1_2')
];
lgraph = addLayers(lgraph, enc1_layers);
lgraph = connectLayers(lgraph, 'input', 'conv_e1_1');

lgraph = addLayers(lgraph, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1'));
lgraph = connectLayers(lgraph, 'relu_e1_2', 'pool1');

% --- Encoder Level 2 (256x256) ---
enc2_layers = [
    convolution2dLayer(3, f2, 'Padding', 'same', 'Name', 'conv_e2_1')
    batchNormalizationLayer('Name', 'bn_e2_1')
    reluLayer('Name', 'relu_e2_1')
    convolution2dLayer(3, f2, 'Padding', 'same', 'Name', 'conv_e2_2')
    batchNormalizationLayer('Name', 'bn_e2_2')
    reluLayer('Name', 'relu_e2_2')
];
lgraph = addLayers(lgraph, enc2_layers);
lgraph = connectLayers(lgraph, 'pool1', 'conv_e2_1');

lgraph = addLayers(lgraph, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2'));
lgraph = connectLayers(lgraph, 'relu_e2_2', 'pool2');

% --- Encoder Level 3 (128x128) ---
enc3_layers = [
    convolution2dLayer(3, f3, 'Padding', 'same', 'Name', 'conv_e3_1')
    batchNormalizationLayer('Name', 'bn_e3_1')
    reluLayer('Name', 'relu_e3_1')
    convolution2dLayer(3, f3, 'Padding', 'same', 'Name', 'conv_e3_2')
    batchNormalizationLayer('Name', 'bn_e3_2')
    reluLayer('Name', 'relu_e3_2')
];
lgraph = addLayers(lgraph, enc3_layers);
lgraph = connectLayers(lgraph, 'pool2', 'conv_e3_1');

lgraph = addLayers(lgraph, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3'));
lgraph = connectLayers(lgraph, 'relu_e3_2', 'pool3');

% --- Encoder Level 4 (64x64) ---
enc4_layers = [
    convolution2dLayer(3, f4, 'Padding', 'same', 'Name', 'conv_e4_1')
    batchNormalizationLayer('Name', 'bn_e4_1')
    reluLayer('Name', 'relu_e4_1')
    convolution2dLayer(3, f4, 'Padding', 'same', 'Name', 'conv_e4_2')
    batchNormalizationLayer('Name', 'bn_e4_2')
    reluLayer('Name', 'relu_e4_2')
];
lgraph = addLayers(lgraph, enc4_layers);
lgraph = connectLayers(lgraph, 'pool3', 'conv_e4_1');

lgraph = addLayers(lgraph, maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4'));
lgraph = connectLayers(lgraph, 'relu_e4_2', 'pool4');

% --- Bottleneck Level 5 (32x32) ---
bottleneck_layers = [
    convolution2dLayer(3, f5, 'Padding', 'same', 'Name', 'conv_b_1')
    batchNormalizationLayer('Name', 'bn_b_1')
    reluLayer('Name', 'relu_b_1')
    convolution2dLayer(3, f5, 'Padding', 'same', 'Name', 'conv_b_2')
    batchNormalizationLayer('Name', 'bn_b_2')
    reluLayer('Name', 'relu_b_2')
];
lgraph = addLayers(lgraph, bottleneck_layers);
lgraph = connectLayers(lgraph, 'pool4', 'conv_b_1');

% --- Decoder Level 4 (64x64) ---
lgraph = addLayers(lgraph, transposedConv2dLayer(2, f4, 'Stride', 2, 'Name', 'upconv_4'));
lgraph = connectLayers(lgraph, 'relu_b_2', 'upconv_4');

% Attention Gate 4
lgraph = addLayers(lgraph, convolution2dLayer(1, f4/2, 'Name', 'ag4_wg'));
lgraph = addLayers(lgraph, convolution2dLayer(1, f4/2, 'Name', 'ag4_wx'));
lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'ag4_add'));
lgraph = addLayers(lgraph, reluLayer('Name', 'ag4_relu'));
lgraph = addLayers(lgraph, convolution2dLayer(1, f4, 'Name', 'ag4_psi'));
lgraph = addLayers(lgraph, sigmoidLayer('Name', 'ag4_sig'));
lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', 'ag4_mult'));

lgraph = connectLayers(lgraph, 'upconv_4', 'ag4_wg');
lgraph = connectLayers(lgraph, 'relu_e4_2', 'ag4_wx');
lgraph = connectLayers(lgraph, 'ag4_wg', 'ag4_add/in1');
lgraph = connectLayers(lgraph, 'ag4_wx', 'ag4_add/in2');
lgraph = connectLayers(lgraph, 'ag4_add', 'ag4_relu');
lgraph = connectLayers(lgraph, 'ag4_relu', 'ag4_psi');
lgraph = connectLayers(lgraph, 'ag4_psi', 'ag4_sig');
lgraph = connectLayers(lgraph, 'relu_e4_2', 'ag4_mult/in1');
lgraph = connectLayers(lgraph, 'ag4_sig', 'ag4_mult/in2');

% Concat & Dec4 Conv
lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name', 'concat_4'));
lgraph = connectLayers(lgraph, 'upconv_4', 'concat_4/in1');
lgraph = connectLayers(lgraph, 'ag4_mult', 'concat_4/in2');

dec4_layers = [
    convolution2dLayer(3, f4, 'Padding', 'same', 'Name', 'conv_d4_1')
    batchNormalizationLayer('Name', 'bn_d4_1')
    reluLayer('Name', 'relu_d4_1')
    convolution2dLayer(3, f4, 'Padding', 'same', 'Name', 'conv_d4_2')
    batchNormalizationLayer('Name', 'bn_d4_2')
    reluLayer('Name', 'relu_d4_2')
];
lgraph = addLayers(lgraph, dec4_layers);
lgraph = connectLayers(lgraph, 'concat_4', 'conv_d4_1');

% --- Decoder Level 3 (128x128) ---
lgraph = addLayers(lgraph, transposedConv2dLayer(2, f3, 'Stride', 2, 'Name', 'upconv_3'));
lgraph = connectLayers(lgraph, 'relu_d4_2', 'upconv_3');

% Attention Gate 3
lgraph = addLayers(lgraph, convolution2dLayer(1, f3/2, 'Name', 'ag3_wg'));
lgraph = addLayers(lgraph, convolution2dLayer(1, f3/2, 'Name', 'ag3_wx'));
lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'ag3_add'));
lgraph = addLayers(lgraph, reluLayer('Name', 'ag3_relu'));
lgraph = addLayers(lgraph, convolution2dLayer(1, f3, 'Name', 'ag3_psi'));
lgraph = addLayers(lgraph, sigmoidLayer('Name', 'ag3_sig'));
lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', 'ag3_mult'));

lgraph = connectLayers(lgraph, 'upconv_3', 'ag3_wg');
lgraph = connectLayers(lgraph, 'relu_e3_2', 'ag3_wx');
lgraph = connectLayers(lgraph, 'ag3_wg', 'ag3_add/in1');
lgraph = connectLayers(lgraph, 'ag3_wx', 'ag3_add/in2');
lgraph = connectLayers(lgraph, 'ag3_add', 'ag3_relu');
lgraph = connectLayers(lgraph, 'ag3_relu', 'ag3_psi');
lgraph = connectLayers(lgraph, 'ag3_psi', 'ag3_sig');
lgraph = connectLayers(lgraph, 'relu_e3_2', 'ag3_mult/in1');
lgraph = connectLayers(lgraph, 'ag3_sig', 'ag3_mult/in2');

% Concat & Dec3 Conv
lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name', 'concat_3'));
lgraph = connectLayers(lgraph, 'upconv_3', 'concat_3/in1');
lgraph = connectLayers(lgraph, 'ag3_mult', 'concat_3/in2');

dec3_layers = [
    convolution2dLayer(3, f3, 'Padding', 'same', 'Name', 'conv_d3_1')
    batchNormalizationLayer('Name', 'bn_d3_1')
    reluLayer('Name', 'relu_d3_1')
    convolution2dLayer(3, f3, 'Padding', 'same', 'Name', 'conv_d3_2')
    batchNormalizationLayer('Name', 'bn_d3_2')
    reluLayer('Name', 'relu_d3_2')
];
lgraph = addLayers(lgraph, dec3_layers);
lgraph = connectLayers(lgraph, 'concat_3', 'conv_d3_1');

% --- Decoder Level 2 (256x256) ---
lgraph = addLayers(lgraph, transposedConv2dLayer(2, f2, 'Stride', 2, 'Name', 'upconv_2'));
lgraph = connectLayers(lgraph, 'relu_d3_2', 'upconv_2');

% Attention Gate 2
lgraph = addLayers(lgraph, convolution2dLayer(1, f2/2, 'Name', 'ag2_wg'));
lgraph = addLayers(lgraph, convolution2dLayer(1, f2/2, 'Name', 'ag2_wx'));
lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'ag2_add'));
lgraph = addLayers(lgraph, reluLayer('Name', 'ag2_relu'));
lgraph = addLayers(lgraph, convolution2dLayer(1, f2, 'Name', 'ag2_psi'));
lgraph = addLayers(lgraph, sigmoidLayer('Name', 'ag2_sig'));
lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', 'ag2_mult'));

lgraph = connectLayers(lgraph, 'upconv_2', 'ag2_wg');
lgraph = connectLayers(lgraph, 'relu_e2_2', 'ag2_wx');
lgraph = connectLayers(lgraph, 'ag2_wg', 'ag2_add/in1');
lgraph = connectLayers(lgraph, 'ag2_wx', 'ag2_add/in2');
lgraph = connectLayers(lgraph, 'ag2_add', 'ag2_relu');
lgraph = connectLayers(lgraph, 'ag2_relu', 'ag2_psi');
lgraph = connectLayers(lgraph, 'ag2_psi', 'ag2_sig');
lgraph = connectLayers(lgraph, 'relu_e2_2', 'ag2_mult/in1');
lgraph = connectLayers(lgraph, 'ag2_sig', 'ag2_mult/in2');

% Concat & Dec2 Conv
lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name', 'concat_2'));
lgraph = connectLayers(lgraph, 'upconv_2', 'concat_2/in1');
lgraph = connectLayers(lgraph, 'ag2_mult', 'concat_2/in2');

dec2_layers = [
    convolution2dLayer(3, f2, 'Padding', 'same', 'Name', 'conv_d2_1')
    batchNormalizationLayer('Name', 'bn_d2_1')
    reluLayer('Name', 'relu_d2_1')
    convolution2dLayer(3, f2, 'Padding', 'same', 'Name', 'conv_d2_2')
    batchNormalizationLayer('Name', 'bn_d2_2')
    reluLayer('Name', 'relu_d2_2')
];
lgraph = addLayers(lgraph, dec2_layers);
lgraph = connectLayers(lgraph, 'concat_2', 'conv_d2_1');

% --- Decoder Level 1 (512x512) ---
lgraph = addLayers(lgraph, transposedConv2dLayer(2, f1, 'Stride', 2, 'Name', 'upconv_1'));
lgraph = connectLayers(lgraph, 'relu_d2_2', 'upconv_1');

% Attention Gate 1
lgraph = addLayers(lgraph, convolution2dLayer(1, f1/2, 'Name', 'ag1_wg'));
lgraph = addLayers(lgraph, convolution2dLayer(1, f1/2, 'Name', 'ag1_wx'));
lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'ag1_add'));
lgraph = addLayers(lgraph, reluLayer('Name', 'ag1_relu'));
lgraph = addLayers(lgraph, convolution2dLayer(1, f1, 'Name', 'ag1_psi'));
lgraph = addLayers(lgraph, sigmoidLayer('Name', 'ag1_sig'));
lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', 'ag1_mult'));

lgraph = connectLayers(lgraph, 'upconv_1', 'ag1_wg');
lgraph = connectLayers(lgraph, 'relu_e1_2', 'ag1_wx');
lgraph = connectLayers(lgraph, 'ag1_wg', 'ag1_add/in1');
lgraph = connectLayers(lgraph, 'ag1_wx', 'ag1_add/in2');
lgraph = connectLayers(lgraph, 'ag1_add', 'ag1_relu');
lgraph = connectLayers(lgraph, 'ag1_relu', 'ag1_psi');
lgraph = connectLayers(lgraph, 'ag1_psi', 'ag1_sig');
lgraph = connectLayers(lgraph, 'relu_e1_2', 'ag1_mult/in1');
lgraph = connectLayers(lgraph, 'ag1_sig', 'ag1_mult/in2');

% Concat & Dec1 Conv
lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name', 'concat_1'));
lgraph = connectLayers(lgraph, 'upconv_1', 'concat_1/in1');
lgraph = connectLayers(lgraph, 'ag1_mult', 'concat_1/in2');

dec1_layers = [
    convolution2dLayer(3, f1, 'Padding', 'same', 'Name', 'conv_d1_1')
    batchNormalizationLayer('Name', 'bn_d1_1')
    reluLayer('Name', 'relu_d1_1')
    convolution2dLayer(3, f1, 'Padding', 'same', 'Name', 'conv_d1_2')
    batchNormalizationLayer('Name', 'bn_d1_2')
    reluLayer('Name', 'relu_d1_2')
    convolution2dLayer(1, numClasses, 'Name', 'conv_out')
    sigmoidLayer('Name', 'sigmoid_out')
];
lgraph = addLayers(lgraph, dec1_layers);
lgraph = connectLayers(lgraph, 'concat_1', 'conv_d1_1');

net = dlnetwork(lgraph);
end
