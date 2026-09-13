"""Reconstructed PyTorch model for MATLAB netV2 EfficientNet-B0 DAGNetwork.

Loads 100% of weights, biases, and BatchNorm statistics from artifacts/matlab_netv2_full_weights.mat
without using stock torchvision weights or torchvision.models.efficientnet_b0 backbone.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATLAB_FULL_WEIGHTS_PATH = PROJECT_ROOT / "artifacts" / "matlab_netv2_full_weights.mat"

# Saved MATLAB netV2 input-layer normalization (RGB scale 0..255)
MATLAB_INPUT_MEAN = np.array([119.57079, 68.79629, 31.649105], dtype=np.float32)
MATLAB_INPUT_STD = np.array([72.80615, 42.005898, 25.419216], dtype=np.float32)


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
            # MATLAB shape [H, W, 1, 1, C] or [H, W, C] -> PyTorch [C, 1, H, W]
            if w_np.ndim == 3:
                w_pt = np.transpose(w_np, (2, 0, 1))[:, None, :, :]
            elif w_np.ndim == 4:
                w_pt = np.transpose(w_np, (3, 2, 0, 1))
            else:
                w_pt = w_np
        else:
            # MATLAB shape [H, W, C_in, C_out] -> PyTorch [C_out, C_in, H, W]
            w_pt = np.transpose(w_np, (3, 2, 0, 1))
            
        with torch.no_grad():
            conv_layer.conv.weight.copy_(torch.from_numpy(w_pt).float())

    if b is not None and conv_layer.conv.bias is not None:
        with torch.no_grad():
            conv_layer.conv.bias.copy_(torch.from_numpy(b.squeeze()).float())


class MatlabNetV2Reconstructed(nn.Module):
    """Complete 1-to-1 PyTorch reconstruction of MATLAB netV2 DAGNetwork."""

    def __init__(self, full_mat_path: Path | str = MATLAB_FULL_WEIGHTS_PATH):
        super().__init__()
        self.full_mat_path = Path(full_mat_path)
        if not self.full_mat_path.is_file():
            raise FileNotFoundError(f"MATLAB weights file not found: {self.full_mat_path}")

        mat_data = scipy.io.loadmat(str(self.full_mat_path))
        layer_names = [str(x[0]) if len(x) > 0 else "" for x in mat_data["layerNames"].squeeze()]

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

        # Construct exact sequential layers
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

        # 3. Head & Classifier
        self.head_conv = TFConv2d(320, 1280, kernel_size=1, stride=1, bias=True)
        self.head_bn = nn.BatchNorm2d(1280, eps=0.001)
        self.head_swish = MatlabSwish()
        self.head_gap = nn.AdaptiveAvgPool2d((1, 1))

        load_matlab_conv(self.head_conv, weights_dict, "efficientnet-b0|model|head|conv2d|Conv2D")
        load_matlab_bn(self.head_bn, weights_dict, "efficientnet-b0|model|head|tpu_batch_normalization|FusedBatchNorm")

        self.dr_head = nn.Linear(1280, 5, bias=True)
        w_head = weights_dict["weights"]["dr_head"]
        b_head = weights_dict["biases"]["dr_head"]
        with torch.no_grad():
            self.dr_head.weight.copy_(torch.from_numpy(w_head).float())
            self.dr_head.bias.copy_(torch.from_numpy(b_head.squeeze()).float())

        # Save weights dict reference for dynamic block building
        self._weights_dict = weights_dict
        self._build_all_blocks()

    def _build_block(self, prefix: str, in_c: int, exp_c: int, out_c: int, k: int, s: int, se_c: int, c_se1: str, c_se2: str) -> nn.ModuleDict:
        """Helper to build standard MBConv block layers matching MATLAB DAGNetwork."""
        block = nn.ModuleDict()
        
        # 1. Expand Conv + BN + Swish
        block["conv_exp"] = TFConv2d(in_c, exp_c, kernel_size=1, stride=1, bias=True)
        block["bn_exp"] = nn.BatchNorm2d(exp_c, eps=0.001)
        block["swish_exp"] = MatlabSwish()
        load_matlab_conv(block["conv_exp"], self._weights_dict, f"{prefix}|conv2d|Conv2D")
        load_matlab_bn(block["bn_exp"], self._weights_dict, f"{prefix}|tpu_batch_normalization|FusedBatchNorm")

        # 2. Depthwise Conv + BN + Swish
        block["conv_dw"] = TFConv2d(exp_c, exp_c, kernel_size=k, stride=s, groups=exp_c, bias=True)
        block["bn_dw"] = nn.BatchNorm2d(exp_c, eps=0.001)
        block["swish_dw"] = MatlabSwish()
        load_matlab_conv(block["conv_dw"], self._weights_dict, f"{prefix}|depthwise_conv2d|depthwise", is_depthwise=True)
        load_matlab_bn(block["bn_dw"], self._weights_dict, f"{prefix}|tpu_batch_normalization_1|FusedBatchNorm")

        # 3. SE Block
        block["se_conv1"] = TFConv2d(exp_c, se_c, kernel_size=1, stride=1, bias=True)
        block["se_swish"] = MatlabSwish()
        block["se_conv2"] = TFConv2d(se_c, exp_c, kernel_size=1, stride=1, bias=True)
        load_matlab_conv(block["se_conv1"], self._weights_dict, c_se1)
        load_matlab_conv(block["se_conv2"], self._weights_dict, c_se2)

        # 4. Project Conv + BN
        block["conv_pw"] = TFConv2d(exp_c, out_c, kernel_size=1, stride=1, bias=True)
        block["bn_pw"] = nn.BatchNorm2d(out_c, eps=0.001)
        load_matlab_conv(block["conv_pw"], self._weights_dict, f"{prefix}|conv2d_1|Conv2D")
        load_matlab_bn(block["bn_pw"], self._weights_dict, f"{prefix}|tpu_batch_normalization_2|FusedBatchNorm")

        return block

    def _build_all_blocks(self) -> None:
        """Build blocks 1 to 15 matching MATLAB DAGNetwork topology."""
        self.blocks = nn.ModuleList()
        
        configs = [
            # Block 1
            ("efficientnet-b0|model|blocks_1", 16, 96, 24, 3, 2, 4, "Conv__309", "Conv__312", False),
            # Block 2
            ("efficientnet-b0|model|blocks_2", 24, 144, 24, 3, 1, 6, "Conv__319", "Conv__322", True),
            # Block 3
            ("efficientnet-b0|model|blocks_3", 24, 144, 40, 5, 2, 6, "Conv__327", "Conv__330", False),
            # Block 4
            ("efficientnet-b0|model|blocks_4", 40, 240, 40, 5, 1, 10, "Conv__337", "Conv__340", True),
            # Block 5
            ("efficientnet-b0|model|blocks_5", 40, 240, 80, 3, 2, 10, "Conv__345", "Conv__348", False),
            # Block 6
            ("efficientnet-b0|model|blocks_6", 80, 480, 80, 3, 1, 20, "Conv__355", "Conv__358", True),
            # Block 7
            ("efficientnet-b0|model|blocks_7", 80, 480, 80, 3, 1, 20, "Conv__365", "Conv__368", True),
            # Block 8
            ("efficientnet-b0|model|blocks_8", 80, 480, 112, 5, 1, 20, "Conv__373", "Conv__376", False),
            # Block 9
            ("efficientnet-b0|model|blocks_9", 112, 672, 112, 5, 1, 28, "Conv__383", "Conv__386", True),
            # Block 10
            ("efficientnet-b0|model|blocks_10", 112, 672, 112, 5, 1, 28, "Conv__393", "Conv__396", True),
            # Block 11
            ("efficientnet-b0|model|blocks_11", 112, 672, 192, 5, 2, 28, "Conv__401", "Conv__404", False),
            # Block 12
            ("efficientnet-b0|model|blocks_12", 192, 1152, 192, 5, 1, 48, "Conv__411", "Conv__414", True),
            # Block 13
            ("efficientnet-b0|model|blocks_13", 192, 1152, 192, 5, 1, 48, "Conv__421", "Conv__424", True),
            # Block 14
            ("efficientnet-b0|model|blocks_14", 192, 1152, 192, 5, 1, 48, "Conv__431", "Conv__434", True),
            # Block 15
            ("efficientnet-b0|model|blocks_15", 192, 1152, 320, 3, 1, 48, "Conv__439", "Conv__442", False),
        ]

        self.block_skips = []
        for cfg in configs:
            prefix, in_c, exp_c, out_c, k, s, se_c, c_se1, c_se2, use_skip = cfg
            b = self._build_block(prefix, in_c, exp_c, out_c, k, s, se_c, c_se1, c_se2)
            self.blocks.append(b)
            self.block_skips.append(use_skip)

    def forward_mbblock(self, x: torch.Tensor, block: nn.ModuleDict, use_skip: bool) -> torch.Tensor:
        """Forward pass through one MBConv block."""
        identity = x
        
        # 1. Expand
        x = block["conv_exp"](x)
        x = block["bn_exp"](x)
        x = block["swish_exp"](x)

        # 2. Depthwise
        x = block["conv_dw"](x)
        x = block["bn_dw"](x)
        x = block["swish_dw"](x)

        # 3. SE Gating
        se = F.adaptive_avg_pool2d(x, (1, 1))
        se = block["se_conv1"](se)
        se = block["se_swish"](se)
        se = block["se_conv2"](se)
        se = torch.sigmoid(se)
        x = x * se

        # 4. Project
        x = block["conv_pw"](x)
        x = block["bn_pw"](x)

        # 5. Skip connection
        if use_skip:
            x = x + identity

        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor."""
        # 1. Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_swish(x)

        # 2. Block 0
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

        # 3. Blocks 1 to 15
        for block, use_skip in zip(self.blocks, self.block_skips):
            x = self.forward_mbblock(x, block, use_skip)

        # 4. Head
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_swish(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass returning 5-class logits."""
        feat = self.forward_features(x)
        pooled = self.head_gap(feat)
        flat = torch.flatten(pooled, 1)
        logits = self.dr_head(flat)
        return logits

    def preprocess_image(self, rgb_image_np: np.ndarray) -> torch.Tensor:
        """Preprocess RGB image uint8 array [H, W, 3] to PyTorch tensor [1, 3, 224, 224]."""
        if rgb_image_np.shape[0] != 224 or rgb_image_np.shape[1] != 224:
            pil_img = Image.fromarray(rgb_image_np).resize((224, 224), resample=Image.Resampling.BILINEAR)
            rgb_image_np = np.array(pil_img)

        img_float = rgb_image_np.astype(np.float32)
        norm_img = (img_float - MATLAB_INPUT_MEAN) / MATLAB_INPUT_STD
        tensor = torch.from_numpy(np.transpose(norm_img, (2, 0, 1))).unsqueeze(0).float()
        return tensor
