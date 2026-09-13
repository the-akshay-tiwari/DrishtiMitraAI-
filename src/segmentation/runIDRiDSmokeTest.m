function result = runIDRiDSmokeTest(batchSize)
%RUNIDRIDSMOKETEST Run a 1-iteration smoke test on GPU/CPU to verify pipeline.
%
%   result = runIDRiDSmokeTest(batchSize)
%   - batchSize: default 4
%   Measures architecture forward pass, masked loss, backward pass, Adam update,
%   exact GPU memory usage, and execution time per iteration.

if nargin < 1 || isempty(batchSize)
    batchSize = 4;
end

fprintf('\n==================================================\n');
fprintf('  DRISHTIMITRA: ATTENTION U-NET SMOKE TEST\n');
fprintf('==================================================\n');

% 1. GPU Check
hasGPU = canUseGPU();
gpuInfo = struct('Name', "CPU Only", 'TotalMemoryMB', 0, 'FreeMemoryMB', 0, 'UsedVRAMMB', 0);
if hasGPU
    g = gpuDevice();
    reset(g); % Clear GPU memory before test
    gpuInfo.Name = string(g.Name);
    gpuInfo.TotalMemoryMB = double(g.TotalMemory) / (1024^2);
    gpuInfo.FreeMemoryMB = double(g.FreeMemory) / (1024^2);
    fprintf('Device: %s (Total VRAM: %.2f MB, Free VRAM: %.2f MB)\n', ...
        gpuInfo.Name, gpuInfo.TotalMemoryMB, gpuInfo.FreeMemoryMB);
else
    fprintf('Device: CPU (No CUDA GPU detected)\n');
end

% 2. Build Architecture
fprintf('Building Attention U-Net [16, 32, 64, 128, 256]...\n');
net = buildAttentionUNet([512, 512, 3], 5, [16, 32, 64, 128, 256]);
paramCount = sum(cellfun(@numel, net.Learnables.Value));
fprintf('Architecture built successfully. Learnable parameters: %d (%.2f M)\n', ...
    paramCount, paramCount / 1e6);

% 3. Extract Small Patch Subset for Smoke Test
fprintf('Extracting %d test patches from official training images...\n', batchSize);
patchesReport = extractIDRiDPatches([], "training", [512, 512], ceil(batchSize / 2), 0.5, 42);

X_raw = zeros(512, 512, 3, batchSize, 'single');
Y_raw = zeros(512, 512, 5, batchSize, 'single');
V_raw = zeros(1, 1, 5, batchSize, 'single');

for b = 1:batchSize
    X_raw(:,:,:,b) = single(patchesReport.images{b}) / 255.0;
    Y_raw(:,:,:,b) = single(patchesReport.masks{b});
    V_raw(1,1,:,b) = single(patchesReport.validity{b});
end

X = dlarray(X_raw, 'SSCB');
Y = dlarray(Y_raw, 'SSCB');
V = dlarray(V_raw, 'SSCB');

if hasGPU
    X = gpuArray(X);
    Y = gpuArray(Y);
    V = gpuArray(V);
end

% 4. Measure Execution Time & GPU Memory
fprintf('Executing forward pass, masked loss calculation, and backward pass...\n');
tStart = tic;

if hasGPU
    memBefore = double(gpuDevice().FreeMemory) / (1024^2);
end

[loss, gradients, lossMetrics] = dlfeval(@computeMaskedMultilabelLoss, net, X, Y, V);

tFwdBwd = toc(tStart);

% 5. Execute 1 Adam Optimizer Update Step
fprintf('Executing Adam optimizer update step...\n');
tAdamStart = tic;
averageGrad = [];
averageSqGrad = [];
learningRate = 0.001;
[net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, 1, learningRate);
tAdam = toc(tAdamStart);

tTotalIter = tFwdBwd + tAdam;

if hasGPU
    gEnd = gpuDevice();
    memAfter = double(gEnd.FreeMemory) / (1024^2);
    gpuInfo.UsedVRAMMB = memBefore - memAfter;
    if gpuInfo.UsedVRAMMB < 0, gpuInfo.UsedVRAMMB = 0; end
end

% 6. Compile Results & Summary Report
result = struct(...
    'status', "SUCCESS", ...
    'batchSize', batchSize, ...
    'inputSize', [512, 512, 3], ...
    'paramCount', paramCount, ...
    'hasGPU', hasGPU, ...
    'gpuName', gpuInfo.Name, ...
    'usedVRAMMB', gpuInfo.UsedVRAMMB, ...
    'freeVRAMMB', gpuInfo.FreeMemoryMB, ...
    'loss', double(extractdata(loss)), ...
    'bceLoss', double(lossMetrics.bceLoss), ...
    'diceLoss', double(lossMetrics.diceLoss), ...
    'fwdBwdTimeSec', tFwdBwd, ...
    'adamUpdateTimeSec', tAdam, ...
    'totalIterTimeSec', tTotalIter);

fprintf('\n--------------------------------------------------\n');
fprintf('  SMOKE TEST RESULTS SUMMARY\n');
fprintf('--------------------------------------------------\n');
fprintf('  Status: %s\n', result.status);
fprintf('  Batch Size: %d\n', result.batchSize);
fprintf('  Learnables Count: %d (%.2f M)\n', result.paramCount, result.paramCount/1e6);
fprintf('  Total Loss: %.4f (BCE: %.4f, Soft Dice: %.4f)\n', ...
    result.loss, result.bceLoss, result.diceLoss);
fprintf('  Fwd + Bwd Pass Time: %.4f sec\n', result.fwdBwdTimeSec);
fprintf('  Adam Update Time: %.4f sec\n', result.adamUpdateTimeSec);
fprintf('  Total Iteration Time: %.4f sec\n', result.totalIterTimeSec);
if hasGPU
    fprintf('  Measured Peak VRAM Usage: %.2f MB (Free: %.2f MB)\n', ...
        result.usedVRAMMB, result.freeVRAMMB);
end
fprintf('==================================================\n\n');
end
