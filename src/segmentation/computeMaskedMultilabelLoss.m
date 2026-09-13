function [loss, gradients, state, lossMetrics] = computeMaskedMultilabelLoss(net, X, Y, V)
%COMPUTEMASKEDMULTILABELLOSS Masked Focal BCE + Tversky Loss for multilabel retinal lesion segmentation.
%
%   [loss, gradients, state, lossMetrics] = computeMaskedMultilabelLoss(net, X, Y, V)
%   - net: dlnetwork
%   - X: dlarray formatted 'SSCB', size [H, W, 3, B]
%   - Y: dlarray formatted 'SSCB', size [H, W, 5, B] (ground truth logical/single)
%   - V: dlarray validity mask, size [1, 1, 5, B] or [5, B] (1 if annotated, 0 if missing)

epsVal = single(1e-7);
smoothVal = single(1.0);

% Ensure V is formatted as [1, 1, 5, B]
[~, ~, ~, B] = size(Y);
V = reshape(V, [1, 1, 5, B]);

% Forward pass with state output for BatchNorm tracking
[Y_pred, state] = forward(net, X);

% Clamp predictions for numerical stability
Y_pred_clamped = max(epsVal, min(single(1.0) - epsVal, Y_pred));

% Channel-specific positive weighting alpha factor:
% 0.75 for sparse lesions (MA, HE, EX, SE), 0.50 for anatomical OD
alpha = dlarray(reshape(single([0.75, 0.75, 0.75, 0.75, 0.50]), [1, 1, 5, 1]), 'SSCB');
gamma = single(2.0);

% 1. Masked Focal Binary Cross Entropy (Focal BCE)
% Down-weights easy background pixels (y=0, y_pred~0 => y_pred^2 ~ 0)
% Boosts positive lesion pixel loss (y=1, alpha=0.75)
pos_focal = -alpha .* ((1 - Y_pred_clamped).^gamma) .* log(Y_pred_clamped);
neg_focal = -(1 - alpha) .* (Y_pred_clamped.^gamma) .* log(1 - Y_pred_clamped);
focal_bce = Y .* pos_focal + (1 - Y) .* neg_focal;

% Average over spatial dimensions [1, 2] -> [1, 1, 5, B]
focal_bce_per_channel = mean(focal_bce, [1 2]);

% Apply validity mask V
masked_focal_bce = focal_bce_per_channel .* V;
valid_v_count = sum(V, 'all');

if valid_v_count > 0
    bce_loss = sum(masked_focal_bce, 'all') / valid_v_count;
else
    bce_loss = sum(masked_focal_bce, 'all');
end

% 2. Masked Focal Tversky Loss (alpha_t = 0.3 for FP, beta_t = 0.7 for FN)
alpha_t = single(0.3);
beta_t = single(0.7);

% Compute TP, FP, FN per channel per batch element with validity masking
Y_valid = Y .* V;
Y_pred_valid = Y_pred_clamped .* V;

TP = sum(Y_valid .* Y_pred_valid, [1 2]);       % [1, 1, 5, B]
FP = sum((1 - Y_valid) .* Y_pred_valid, [1 2]); % [1, 1, 5, B]
FN = sum(Y_valid .* (1 - Y_pred_valid), [1 2]); % [1, 1, 5, B]

tversky = (TP + smoothVal) ./ (TP + alpha_t * FP + beta_t * FN + smoothVal);
tversky_loss_per_channel = 1 - tversky; % [1, 1, 5, B]

% Apply validity mask to Tversky loss
masked_tversky = tversky_loss_per_channel .* V;

if valid_v_count > 0
    dice_loss = sum(masked_tversky, 'all') / valid_v_count;
else
    dice_loss = single(0);
end

% 3. Total Loss (0.5 Focal BCE + 0.5 Tversky Loss)
loss = 0.5 * bce_loss + 0.5 * dice_loss;

% Compute gradients
gradients = dlgradient(loss, net.Learnables);

% Metrics for logging (extract raw values)
if nargout >= 4
    lossMetrics = struct(...
        'totalLoss', extractdata(loss), ...
        'bceLoss', extractdata(bce_loss), ...
        'diceLoss', extractdata(dice_loss));
end
end
