"""Reconstructed PyTorch model for MATLAB netV3_4 CORAL Ordinal Classifier.

Loads 100% of weights, biases, and BatchNorm statistics from
artifacts_output/v3_4/models/matlab_v3_4_full_weights.mat
without using stock torchvision weights or torchvision.models.efficientnet_b0 backbone.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import scipy.io
import skimage.color as skcolor
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V3_4_WEIGHTS_PATH = (
    PROJECT_ROOT / "artifacts_output" / "v3_4" / "models" / "matlab_v3_4_full_weights.mat"
)

# Saved MATLAB netV3_4 input-layer normalization (RGB scale 0..255)
V3_4_INPUT_MEAN = np.array([118.8702, 68.16972, 31.435555], dtype=np.float32)
V3_4_INPUT_STD = np.array([72.033966, 41.878704, 25.318834], dtype=np.float32)


# =====================================================================
# 1. MATLAB-Matched Preprocessing (LAB CLAHE)
# =====================================================================

def matlab_clip_histogram(img_hist: np.ndarray, clip_limit: int, num_bins: int) -> np.ndarray:
    """Exact replica of MATLAB's clipHistogram internal function in adapthisteq.m."""
    hist = img_hist.copy()
    total_excess = int(np.sum(np.maximum(hist - clip_limit, 0)))
    avg_bin_incr = total_excess // num_bins
    upper_limit = clip_limit - avg_bin_incr

    for k in range(num_bins):
        if hist[k] > clip_limit:
            hist[k] = clip_limit
        else:
            if hist[k] > upper_limit:
                total_excess -= (clip_limit - hist[k])
                hist[k] = clip_limit
            else:
                total_excess -= avg_bin_incr
                hist[k] += avg_bin_incr

    k = 0
    while total_excess > 0:
        step_size = max(num_bins // total_excess, 1)
        for m in range(k, num_bins, step_size):
            if hist[m] < clip_limit:
                hist[m] += 1
                total_excess -= 1
                if total_excess == 0:
                    break
        k += 1
        if k >= num_bins:
            k = 0
    return hist


def matlab_adapthisteq_exact(
    I: np.ndarray,
    num_tiles: tuple[int, int] = (8, 8),
    norm_clip_limit: float = 0.01,
    num_bins: int = 256,
) -> np.ndarray:
    """Exact 1-to-1 NumPy port of MATLAB's adapthisteq.m for 2D float image in [0, 1]."""
    h, w = I.shape
    num_tiles_r, num_tiles_c = num_tiles
    dim_tile_r = h // num_tiles_r
    dim_tile_c = w // num_tiles_c

    num_pix_in_tile = dim_tile_r * dim_tile_c
    min_clip_limit = int(np.ceil(num_pix_in_tile / num_bins))
    clip_limit = int(min_clip_limit + np.round(norm_clip_limit * (num_pix_in_tile - min_clip_limit)))

    # 1. Tile mappings
    tile_mappings = np.zeros((num_tiles_r, num_tiles_c, num_bins), dtype=np.float64)

    for r in range(num_tiles_r):
        for c in range(num_tiles_c):
            tile = I[r * dim_tile_r : (r + 1) * dim_tile_r, c * dim_tile_c : (c + 1) * dim_tile_c]
            bin_idx = np.clip(np.floor(tile * (num_bins - 1) + 0.5).astype(int), 0, num_bins - 1)
            hist = np.bincount(bin_idx.flatten(), minlength=num_bins)
            hist = matlab_clip_histogram(hist, clip_limit, num_bins)
            mapping = np.cumsum(hist) / num_pix_in_tile
            tile_mappings[r, c] = np.clip(mapping, 0.0, 1.0)

    # 2. Bilinear interpolation across tiles
    out = np.zeros((h, w), dtype=np.float64)
    img_pix_vals = np.clip(np.floor(I * (num_bins - 1) + 0.5).astype(int), 0, num_bins - 1)

    img_tile_row = 0
    for k in range(num_tiles_r + 1):
        if k == 0:
            num_rows = dim_tile_r // 2
            map_r0, map_r1 = 0, 0
        elif k == num_tiles_r:
            num_rows = dim_tile_r // 2
            map_r0, map_r1 = num_tiles_r - 1, num_tiles_r - 1
        else:
            num_rows = dim_tile_r
            map_r0, map_r1 = k - 1, k

        img_tile_col = 0
        for l in range(num_tiles_c + 1):
            if l == 0:
                num_cols = dim_tile_c // 2
                map_c0, map_c1 = 0, 0
            elif l == num_tiles_c:
                num_cols = dim_tile_c // 2
                map_c0, map_c1 = num_tiles_c - 1, num_tiles_c - 1
            else:
                num_cols = dim_tile_c
                map_c0, map_c1 = l - 1, l

            ul = tile_mappings[map_r0, map_c0]
            ur = tile_mappings[map_r0, map_c1]
            bl = tile_mappings[map_r1, map_c0]
            br = tile_mappings[map_r1, map_c1]

            sub_pix = img_pix_vals[
                img_tile_row : img_tile_row + num_rows,
                img_tile_col : img_tile_col + num_cols,
            ]

            v_ul = ul[sub_pix]
            v_ur = ur[sub_pix]
            v_bl = bl[sub_pix]
            v_br = br[sub_pix]

            row_idx = np.arange(num_rows).reshape(num_rows, 1)
            col_idx = np.arange(num_cols).reshape(1, num_cols)
            row_rev = (num_rows - np.arange(num_rows)).reshape(num_rows, 1)
            col_rev = (num_cols - np.arange(num_cols)).reshape(1, num_cols)

            norm_factor = float(num_rows * num_cols)
            val = (
                row_rev * (col_rev * v_ul + col_idx * v_ur)
                + row_idx * (col_rev * v_bl + col_idx * v_br)
            ) / norm_factor
            out[img_tile_row : img_tile_row + num_rows, img_tile_col : img_tile_col + num_cols] = val
            img_tile_col += num_cols
        img_tile_row += num_rows

    return out


def enhance_image_v2(rgb_uint8: np.ndarray) -> np.ndarray:
    """Exact Python equivalent of MATLAB's enhanceImageV2: LAB CLAHE on L-channel."""
    lab = skcolor.rgb2lab(rgb_uint8)
    L = lab[:, :, 0] / 100.0
    L_eq = matlab_adapthisteq_exact(L, num_tiles=(8, 8), norm_clip_limit=0.01, num_bins=256)
    lab[:, :, 0] = L_eq * 100.0
    rgb_float = skcolor.lab2rgb(lab)
    rgb_out = np.clip(np.round(rgb_float * 255.0), 0, 255).astype(np.uint8)
    return rgb_out


# =====================================================================
# 2. Neural Network Building Blocks
# =====================================================================

class TFConv2d(nn.Module):
    """Conv2d with TensorFlow/MATLAB 'same' padding logic."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1 and self.kernel_size == 1:
            return self.conv(x)

        in_h, in_w = x.shape[2], x.shape[3]
        out_h = int(np.ceil(float(in_h) / float(self.stride)))
        out_w = int(np.ceil(float(in_w) / float(self.stride)))

        pad_h = max((out_h - 1) * self.stride + self.kernel_size - in_h, 0)
        pad_w = max((out_w - 1) * self.stride + self.kernel_size - in_w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


class MatlabSwish(nn.Module):
    """MATLAB TPU Swish activation: x * sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def load_matlab_bn(bn_layer: nn.BatchNorm2d, weights_dict: dict, name: str) -> None:
    """Load MATLAB TrainedMean, TrainedVariance, Scale, Offset into PyTorch BatchNorm2d."""
    tm = weights_dict["means"].get(name)
    tv = weights_dict["vars"].get(name)
    sc = weights_dict["weights"].get(name)
    off = weights_dict["biases"].get(name)

    with torch.no_grad():
        if tm is not None:
            bn_layer.running_mean.copy_(torch.from_numpy(tm.squeeze()).float())
        if tv is not None:
            bn_layer.running_var.copy_(torch.from_numpy(tv.squeeze()).float())
        if sc is not None and len(sc) > 0:
            bn_layer.weight.copy_(torch.from_numpy(sc.squeeze()).float())
        else:
            bn_layer.weight.data.fill_(1.0)
        if off is not None and len(off) > 0:
            bn_layer.bias.copy_(torch.from_numpy(off.squeeze()).float())
        else:
            bn_layer.bias.data.fill_(0.0)


def load_matlab_conv(conv_layer: TFConv2d, weights_dict: dict, name: str, is_depthwise: bool = False) -> None:
    """Load MATLAB Conv2D/GroupedConv2D weights & biases into PyTorch TFConv2d."""
    w = weights_dict["weights"].get(name)
    b = weights_dict["biases"].get(name)

    if w is not None:
        w_np = np.squeeze(w) if is_depthwise else w
        if is_depthwise:
            if w_np.ndim == 3:
                w_pt = np.transpose(w_np, (2, 0, 1))[:, None, :, :]
            elif w_np.ndim == 4:
                w_pt = np.transpose(w_np, (3, 2, 0, 1))
            else:
                w_pt = w_np
        else:
            w_pt = np.transpose(w_np, (3, 2, 0, 1))

        with torch.no_grad():
            conv_layer.conv.weight.copy_(torch.from_numpy(w_pt).float())

    if b is not None and conv_layer.conv.bias is not None:
        with torch.no_grad():
            conv_layer.conv.bias.copy_(torch.from_numpy(b.squeeze()).float())


# =====================================================================
# 3. Complete V3.4 Ordinal Classifier Architecture
# =====================================================================

class MatlabNetV3_4Reconstructed(nn.Module):
    """Complete 1-to-1 PyTorch reconstruction of MATLAB netV3_4 CORAL Ordinal Classifier."""

    def __init__(self, full_mat_path: Path | str = DEFAULT_V3_4_WEIGHTS_PATH):
        super().__init__()
        self.full_mat_path = Path(full_mat_path)
        if not self.full_mat_path.is_file():
            raise FileNotFoundError(f"MATLAB weights file not found: {self.full_mat_path}")

        mat_data = scipy.io.loadmat(str(self.full_mat_path))
        layer_names = [str(x[0]) if len(x) > 0 else "" for x in mat_data["layerNames"].squeeze()]

        # Load normalization from mat file
        self.input_mean = mat_data["inputMean"].squeeze().astype(np.float32)
        self.input_std = mat_data["inputStd"].squeeze().astype(np.float32)

        # Build dictionaries for fast layer parameter lookup
        weights_dict = {"weights": {}, "biases": {}, "means": {}, "vars": {}}
        for i, name in enumerate(layer_names):
            w = mat_data["layerWeights"][0, i]
            b = mat_data["layerBiases"][0, i]
            tm = mat_data["layerTrainedMeans"][0, i]
            tv = mat_data["layerTrainedVars"][0, i]
            if w.size > 0:
                weights_dict["weights"][name] = w
            if b.size > 0:
                weights_dict["biases"][name] = b
            if tm.size > 0:
                weights_dict["means"][name] = tm
            if tv.size > 0:
                weights_dict["vars"][name] = tv

        self._weights_dict = weights_dict

        # Construct EfficientNet-B0 backbone
        # 1. Stem
        self.stem_conv = TFConv2d(3, 32, kernel_size=3, stride=2, bias=True)
        self.stem_bn = nn.BatchNorm2d(32, eps=0.001)
        self.stem_swish = MatlabSwish()
        load_matlab_conv(self.stem_conv, weights_dict, "efficientnet-b0|model|stem|conv2d|Conv2D")
        load_matlab_bn(self.stem_bn, weights_dict, "efficientnet-b0|model|stem|tpu_batch_normalization|FusedBatchNorm")

        # 2. Block 0 (MBConv1 k3s1, 32->16, SE)
        self.b0_dw = TFConv2d(32, 32, kernel_size=3, stride=1, groups=32, bias=True)
        self.b0_bn0 = nn.BatchNorm2d(32, eps=0.001)
        self.b0_swish0 = MatlabSwish()
        self.b0_se_conv1 = TFConv2d(32, 8, kernel_size=1, stride=1, bias=True)
        self.b0_se_swish = MatlabSwish()
        self.b0_se_conv2 = TFConv2d(8, 32, kernel_size=1, stride=1, bias=True)
        self.b0_pw = TFConv2d(32, 16, kernel_size=1, stride=1, bias=True)
        self.b0_bn1 = nn.BatchNorm2d(16, eps=0.001)

        load_matlab_conv(self.b0_dw, weights_dict, "efficientnet-b0|model|blocks_0|depthwise_conv2d|depthwise", is_depthwise=True)
        load_matlab_bn(self.b0_bn0, weights_dict, "efficientnet-b0|model|blocks_0|tpu_batch_normalization|FusedBatchNorm")
        load_matlab_conv(self.b0_se_conv1, weights_dict, "Conv__301")
        load_matlab_conv(self.b0_se_conv2, weights_dict, "Conv__304")
        load_matlab_conv(self.b0_pw, weights_dict, "efficientnet-b0|model|blocks_0|conv2d|Conv2D")
        load_matlab_bn(self.b0_bn1, weights_dict, "efficientnet-b0|model|blocks_0|tpu_batch_normalization_1|FusedBatchNorm")

        # 3. Blocks 1 to 15
        self.blocks = nn.ModuleList()
        self._build_all_blocks(weights_dict)

        # 4. Head Convolution & Pooling
        self.head_conv = TFConv2d(320, 1280, kernel_size=1, stride=1, bias=True)
        self.head_bn = nn.BatchNorm2d(1280, eps=0.001)
        self.head_swish = MatlabSwish()
        self.head_gap = nn.AdaptiveAvgPool2d((1, 1))

        load_matlab_conv(self.head_conv, weights_dict, "efficientnet-b0|model|head|conv2d|Conv2D")
        load_matlab_bn(self.head_bn, weights_dict, "efficientnet-b0|model|head|tpu_batch_normalization|FusedBatchNorm")

        # 5. V3.4 CORAL Ordinal Head
        # Shared scalar score projection: w^T g(x)
        self.dr_ordinal_score = nn.Linear(1280, 1, bias=True)
        w_score = weights_dict["weights"]["dr_ordinal_score"]
        b_score = weights_dict["biases"].get("dr_ordinal_score")

        with torch.no_grad():
            self.dr_ordinal_score.weight.copy_(torch.from_numpy(w_score).float())
            if b_score is not None and b_score.size > 0:
                self.dr_ordinal_score.bias.copy_(torch.from_numpy(b_score.squeeze()[None]).float())
            else:
                self.dr_ordinal_score.bias.zero_()

        # 4 Rank Threshold Biases [b1, b2, b3, b4]
        b_thresh = weights_dict["biases"]["dr_ordinal_thresholds"]
        self.dr_ordinal_thresholds_bias = nn.Parameter(
            torch.from_numpy(b_thresh.squeeze()).float(), requires_grad=False
        )

        # Freeze network in evaluation mode by default
        self.eval()

    def _build_block(
        self,
        weights_dict: dict,
        prefix: str,
        in_c: int,
        exp_c: int,
        out_c: int,
        k: int,
        s: int,
        se_c: int,
        c_se1: str,
        c_se2: str,
    ) -> nn.ModuleDict:
        b = nn.ModuleDict(
            {
                "conv_exp": TFConv2d(in_c, exp_c, kernel_size=1, stride=1, bias=True),
                "bn_exp": nn.BatchNorm2d(exp_c, eps=0.001),
                "swish_exp": MatlabSwish(),
                "conv_dw": TFConv2d(exp_c, exp_c, kernel_size=k, stride=s, groups=exp_c, bias=True),
                "bn_dw": nn.BatchNorm2d(exp_c, eps=0.001),
                "swish_dw": MatlabSwish(),
                "se_conv1": TFConv2d(exp_c, se_c, kernel_size=1, stride=1, bias=True),
                "se_swish": MatlabSwish(),
                "se_conv2": TFConv2d(se_c, exp_c, kernel_size=1, stride=1, bias=True),
                "conv_pw": TFConv2d(exp_c, out_c, kernel_size=1, stride=1, bias=True),
                "bn_pw": nn.BatchNorm2d(out_c, eps=0.001),
            }
        )

        load_matlab_conv(b["conv_exp"], weights_dict, f"{prefix}|conv2d|Conv2D")
        load_matlab_bn(b["bn_exp"], weights_dict, f"{prefix}|tpu_batch_normalization|FusedBatchNorm")
        load_matlab_conv(b["conv_dw"], weights_dict, f"{prefix}|depthwise_conv2d|depthwise", is_depthwise=True)
        load_matlab_bn(b["bn_dw"], weights_dict, f"{prefix}|tpu_batch_normalization_1|FusedBatchNorm")
        load_matlab_conv(b["se_conv1"], weights_dict, c_se1)
        load_matlab_conv(b["se_conv2"], weights_dict, c_se2)
        load_matlab_conv(b["conv_pw"], weights_dict, f"{prefix}|conv2d_1|Conv2D")
        load_matlab_bn(b["bn_pw"], weights_dict, f"{prefix}|tpu_batch_normalization_2|FusedBatchNorm")
        return b

    def _build_all_blocks(self, weights_dict: dict) -> None:
        configs = [
            ("efficientnet-b0|model|blocks_1", 16, 96, 24, 3, 2, 4, "Conv__309", "Conv__312", False),
            ("efficientnet-b0|model|blocks_2", 24, 144, 24, 3, 1, 6, "Conv__319", "Conv__322", True),
            ("efficientnet-b0|model|blocks_3", 24, 144, 40, 5, 2, 6, "Conv__327", "Conv__330", False),
            ("efficientnet-b0|model|blocks_4", 40, 240, 40, 5, 1, 10, "Conv__337", "Conv__340", True),
            ("efficientnet-b0|model|blocks_5", 40, 240, 80, 3, 2, 10, "Conv__345", "Conv__348", False),
            ("efficientnet-b0|model|blocks_6", 80, 480, 80, 3, 1, 20, "Conv__355", "Conv__358", True),
            ("efficientnet-b0|model|blocks_7", 80, 480, 80, 3, 1, 20, "Conv__365", "Conv__368", True),
            ("efficientnet-b0|model|blocks_8", 80, 480, 112, 5, 1, 20, "Conv__373", "Conv__376", False),
            ("efficientnet-b0|model|blocks_9", 112, 672, 112, 5, 1, 28, "Conv__383", "Conv__386", True),
            ("efficientnet-b0|model|blocks_10", 112, 672, 112, 5, 1, 28, "Conv__393", "Conv__396", True),
            ("efficientnet-b0|model|blocks_11", 112, 672, 192, 5, 2, 28, "Conv__401", "Conv__404", False),
            ("efficientnet-b0|model|blocks_12", 192, 1152, 192, 5, 1, 48, "Conv__411", "Conv__414", True),
            ("efficientnet-b0|model|blocks_13", 192, 1152, 192, 5, 1, 48, "Conv__421", "Conv__424", True),
            ("efficientnet-b0|model|blocks_14", 192, 1152, 192, 5, 1, 48, "Conv__431", "Conv__434", True),
            ("efficientnet-b0|model|blocks_15", 192, 1152, 320, 3, 1, 48, "Conv__439", "Conv__442", False),
        ]

        self.block_skips = []
        for cfg in configs:
            prefix, in_c, exp_c, out_c, k, s, se_c, c_se1, c_se2, use_skip = cfg
            b = self._build_block(weights_dict, prefix, in_c, exp_c, out_c, k, s, se_c, c_se1, c_se2)
            self.blocks.append(b)
            self.block_skips.append(use_skip)

    def forward_mbblock(self, x: torch.Tensor, block: nn.ModuleDict, use_skip: bool) -> torch.Tensor:
        identity = x
        x = block["conv_exp"](x)
        x = block["bn_exp"](x)
        x = block["swish_exp"](x)

        x = block["conv_dw"](x)
        x = block["bn_dw"](x)
        x = block["swish_dw"](x)

        se = F.adaptive_avg_pool2d(x, (1, 1))
        se = block["se_conv1"](se)
        se = block["se_swish"](se)
        se = block["se_conv2"](se)
        se = torch.sigmoid(se)
        x = x * se

        x = block["conv_pw"](x)
        x = block["bn_pw"](x)

        if use_skip:
            x = x + identity
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_swish(x)

        x = self.b0_dw(x)
        x = self.b0_bn0(x)
        x = self.b0_swish0(x)
        se = F.adaptive_avg_pool2d(x, (1, 1))
        se = self.b0_se_conv1(se)
        se = self.b0_se_swish(se)
        se = self.b0_se_conv2(se)
        se = torch.sigmoid(se)
        x = x * se
        x = self.b0_pw(x)
        x = self.b0_bn1(x)

        for block, use_skip in zip(self.blocks, self.block_skips):
            x = self.forward_mbblock(x, block, use_skip)

        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_swish(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.
        
        Returns:
            probs: Cumulative probabilities [B, 4] in (0, 1)
            score: Shared scalar ordinal score [B, 1]
            logits: Rank threshold logits [B, 4]
        """
        feat = self.forward_features(x)
        pooled = self.head_gap(feat)
        flat = torch.flatten(pooled, 1)
        score = self.dr_ordinal_score(flat)  # [B, 1]
        logits = score + self.dr_ordinal_thresholds_bias  # [B, 4]
        probs = torch.sigmoid(logits)  # [B, 4]
        return probs, score, logits

    def preprocess_image(self, rgb_image_np: np.ndarray) -> torch.Tensor:
        """Preprocess RGB image uint8 array [H, W, 3] to PyTorch tensor [1, 3, 224, 224]."""
        if rgb_image_np.shape[0] != 224 or rgb_image_np.shape[1] != 224:
            pil_img = Image.fromarray(rgb_image_np).resize((224, 224), resample=Image.Resampling.BILINEAR)
            rgb_image_np = np.array(pil_img)

        img_float = rgb_image_np.astype(np.float32)
        norm_img = (img_float - self.input_mean) / self.input_std
        tensor = torch.from_numpy(np.transpose(norm_img, (2, 0, 1))).unsqueeze(0).float()
        return tensor

    def predict_single_image(
        self,
        raw_rgb_np: np.ndarray,
        apply_clahe: bool = True,
    ) -> dict:
        """Run single-image inference end-to-end.
        
        Args:
            raw_rgb_np: uint8 RGB numpy array of arbitrary resolution.
            apply_clahe: If True, applies MATLAB-matched enhanceImageV2 LAB CLAHE.
            
        Returns:
            Dictionary with grades, probabilities, and clinical referral.
        """
        self.eval()
        with torch.no_grad():
            if apply_clahe:
                enhanced = enhance_image_v2(raw_rgb_np)
            else:
                enhanced = raw_rgb_np

            tensor = self.preprocess_image(enhanced)
            device = next(self.parameters()).device
            tensor = tensor.to(device)
            probs, score, logits = self.forward(tensor)

            probs_np = probs.squeeze(0).cpu().numpy()
            score_val = float(score.squeeze().cpu().item())
            logits_np = logits.squeeze(0).cpu().numpy()

            primary_grade = int(np.sum(probs_np > 0.5))
            is_referable = bool(primary_grade >= 2)

            expected_grade = float(np.sum(probs_np))
            val_calibrated_thresholds = [0.40, 1.45, 2.40, 3.25]
            calibrated_grade = int(sum(expected_grade >= t for t in val_calibrated_thresholds))

            grade_names = [
                "No DR (Grade 0)",
                "Mild DR (Grade 1)",
                "Moderate DR (Grade 2)",
                "Severe DR (Grade 3)",
                "Proliferative DR (Grade 4)",
            ]

            return {
                "primary_grade": primary_grade,
                "primary_grade_name": grade_names[primary_grade],
                "is_referable": is_referable,
                "cumulative_probs": {
                    "P_ge_1": float(probs_np[0]),
                    "P_ge_2": float(probs_np[1]),
                    "P_ge_3": float(probs_np[2]),
                    "P_ge_4": float(probs_np[3]),
                },
                "score": score_val,
                "logits": logits_np.tolist(),
                "expected_grade": expected_grade,
                "calibrated_grade": calibrated_grade,
                "calibrated_grade_name": grade_names[calibrated_grade],
                "monotonic": bool(
                    probs_np[0] >= probs_np[1] - 1e-6
                    and probs_np[1] >= probs_np[2] - 1e-6
                    and probs_np[2] >= probs_np[3] - 1e-6
                ),
            }
