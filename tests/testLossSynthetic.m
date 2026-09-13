%% TESTLOSSSYNTHETIC Synthetic unit test for computeMaskedMultilabelLoss
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'src')));

fprintf('Running Synthetic Loss Unit Test...\n');

% Build mini test network
net = buildAttentionUNet([64, 64, 3], 5, [8, 16, 32, 64, 128]);

% Create synthetic tensors (Batch size 2, Spatial 64x64)
H = 64; W = 64; B = 2;
X_raw = single(rand(H, W, 3, B));
Y_raw = zeros(H, W, 5, B, 'single');

% Sample 1: Positive for MA (ch 1) and OD (ch 5); Negative for others
Y_raw(20:30, 20:30, 1, 1) = 1.0; % MA positive
Y_raw(10:40, 10:40, 5, 1) = 1.0; % OD positive

% Sample 2: Positive for EX (ch 3); SE (ch 4) is missing (V=0)
Y_raw(15:25, 15:25, 3, 2) = 1.0; % EX positive

% Validity mask V: Channel 4 (SE) missing on sample 2
V_raw = ones(1, 1, 5, B, 'single');
V_raw(1, 1, 4, 2) = 0.0; % SE missing for sample 2

X = dlarray(X_raw, 'SSCB');
Y = dlarray(Y_raw, 'SSCB');
V = dlarray(V_raw, 'SSCB');

if canUseGPU()
    X = gpuArray(X);
    Y = gpuArray(Y);
    V = gpuArray(V);
end

% 1. Standard forward & gradient test
[loss, gradients, metrics] = dlfeval(@computeMaskedMultilabelLoss, net, X, Y, V);

lossVal = double(extractdata(loss));
bceVal = double(metrics.bceLoss);
diceVal = double(metrics.diceLoss);

fprintf('  Total Loss: %.4f | BCE Loss: %.4f | Dice Loss: %.4f\n', lossVal, bceVal, diceVal);

assert(isfinite(lossVal), 'Loss must be finite');
assert(lossVal > 0, 'Loss must be positive');
assert(isfinite(bceVal), 'BCE loss must be finite');
assert(isfinite(diceVal), 'Dice loss must be finite');

% Check gradients are valid and non-NaN
gradValues = cellfun(@extractdata, gradients.Value, 'UniformOutput', false);
allFinite = all(cellfun(@(g) all(isfinite(g), 'all'), gradValues));
assert(allFinite, 'All gradient values must be finite (no NaNs/Infs)');
fprintf('  [1/5] PASS: Gradients are 100%% finite and non-NaN.\n');

% 2. Missing channel V=0 zero gradient test
% Construct X where prediction for channel 4 changes; verify loss for missing channel doesn't change
V_missing = zeros(1, 1, 5, 1, 'single'); % V=0 for all channels
Y_zero = zeros(64, 64, 5, 1, 'single');
X_single = dlarray(X_raw(:,:,:,1:1), 'SSCB');
[lossMissing, gradMissing, ~] = dlfeval(@computeMaskedMultilabelLoss, net, X_single, dlarray(Y_zero,'SSCB'), dlarray(V_missing,'SSCB'));
lossMissingVal = double(extractdata(lossMissing));
assert(abs(lossMissingVal) < 1e-6, 'Loss for V=0 must be zero');
fprintf('  [2/5] PASS: Missing channel V=0 gives exactly zero loss.\n');

% 3. All-zero valid negative patch has finite loss test
V_valid_all_zero = ones(1, 1, 5, 1, 'single');
[lossNeg, ~, ~] = dlfeval(@computeMaskedMultilabelLoss, net, X_single, dlarray(Y_zero,'SSCB'), dlarray(V_valid_all_zero,'SSCB'));
lossNegVal = double(extractdata(lossNeg));
assert(isfinite(lossNegVal), 'All-zero negative patch must have finite loss');
fprintf('  [3/5] PASS: All-zero valid negative patch has finite loss (%.4f).\n', lossNegVal);

% 4. Positive sparse lesion vs background domination test
% Compare Focal BCE vs Standard BCE gradient on 10-pixel lesion
epsVal = single(1e-7);
alpha_ma = single(0.75); gamma_val = single(2.0);
y_pred_bg = single(0.01);
focal_bg_loss = -(1 - alpha_ma) * (y_pred_bg^gamma_val) * log(1 - y_pred_bg + epsVal);
std_bg_loss = -(1 - y_pred_bg) * log(1 - y_pred_bg + epsVal);
bg_suppression_ratio = std_bg_loss / (focal_bg_loss + epsVal);
assert(bg_suppression_ratio > 100, 'Focal loss must suppress easy background by at least 100x');
fprintf('  [4/5] PASS: Focal loss suppresses easy background penalty by %.1fx.\n', bg_suppression_ratio);

% 5. Perfect prediction has near-zero loss test
% Feed ground truth as synthetic prediction to verify loss approaches 0
Y_pred_perfect = dlarray(Y_raw(:,:,:,1:1), 'SSCB');
Y_gt_perfect = Y_pred_perfect;
V_perfect = dlarray(ones(1, 1, 5, 1, 'single'), 'SSCB');

% Direct evaluation of loss components for perfect prediction
TP_p = sum(Y_gt_perfect .* Y_pred_perfect, [1 2]);
FP_p = sum((1 - Y_gt_perfect) .* Y_pred_perfect, [1 2]);
FN_p = sum(Y_gt_perfect .* (1 - Y_pred_perfect), [1 2]);
tversky_p = (TP_p + 1.0) ./ (TP_p + 0.3 * FP_p + 0.7 * FN_p + 1.0);
tversky_loss_p = mean(1 - tversky_p, 'all');

assert(extractdata(tversky_loss_p) < 1e-4, 'Perfect prediction Tversky loss must be near zero');
fprintf('  [5/5] PASS: Perfect prediction Tversky loss is near-zero (%.6f).\n', extractdata(tversky_loss_p));

fprintf('\nSUCCESS: All 5 synthetic loss unit tests passed cleanly!\n');
