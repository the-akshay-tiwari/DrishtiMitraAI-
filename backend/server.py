"""FastAPI inference service for the trained DrishtiMitra research model."""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision.models import efficientnet_b0
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATLAB_WEIGHTS_PATH = PROJECT_ROOT / "artifacts" / "matlab_dr_classifier_v2_weights.mat"
E02_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E02_no_sampler_35ep" / "checkpoint.pt"
BASELINE_PATH = PROJECT_ROOT / "artifacts" / "drishtimitra_efficientnet_b0.pt"
DIST_DIR = PROJECT_ROOT / "dist"
ASSETS_DIR = DIST_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "v0.2-matlab-netv2-reconstructed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_BYTES = 12 * 1024 * 1024


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [backend] %(message)s",
    handlers=[FlushStreamHandler(sys.stdout)],
)
logger = logging.getLogger("drishtimitra")

app = FastAPI(title="DrishtiMitra experimental inference API", version="0.2.0")

cors_origins_value = os.getenv("CORS_ALLOWED_ORIGINS", "*")
cors_allowed_origins = [origin.strip() for origin in cors_origins_value.split(",") if origin.strip()]
if not cors_allowed_origins:
    cors_allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins if "*" not in cors_allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

model: torch.nn.Module | None = None
class_names: list[str] = ["0", "1", "2", "3", "4"]
transform: v2.Compose | None = None
active_checkpoint_path: str = ""
loaded_experiment_id: str = ""


def get_process_rss_mb() -> float:
    """Returns process Resident Set Size (RSS) in MB using /proc/self/status on Linux or psutil fallback."""
    proc_status = Path("/proc/self/status")
    if proc_status.is_file():
        try:
            with open(proc_status, "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return round(float(parts[1]) / 1024.0, 2)
        except Exception:
            pass

    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except Exception:
        pass

    return 0.0


def get_model_diagnostics(target_model: torch.nn.Module, image_tensor: torch.Tensor) -> dict[str, object]:
    """Returns explicit diagnostic details of target model and input tensor."""
    param_count = sum(p.numel() for p in target_model.parameters())
    first_param_device = next(target_model.parameters()).device
    return {
        "model_class": type(target_model).__name__,
        "model_device": str(first_param_device),
        "model_eval_mode": not target_model.training,
        "tensor_shape": list(image_tensor.shape),
        "tensor_dtype": str(image_tensor.dtype),
        "tensor_device": str(image_tensor.device),
        "parameter_count": param_count,
    }


def pad_to_square(image: Image.Image, bg_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Pad image with black borders to make it 1:1 square without distorting aspect ratio."""
    width, height = image.size
    if width == height:
        return image
    max_dim = max(width, height)
    new_img = Image.new(image.mode, (max_dim, max_dim), bg_color)
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    new_img.paste(image, (paste_x, paste_y))
    return new_img


def validate_retina_image(image: Image.Image) -> tuple[bool, str]:
    """Validate whether an image is a valid, clear fundus retina image."""
    w, h = image.size
    rss_val_start = get_process_rss_mb()
    logger.info(f"[PREDICT] stage=validation_downsample_start | orig_dimensions={w}x{h} | rss_mb={rss_val_start}MB")
    sys.stdout.flush()

    if w > 1024 or h > 1024:
        val_img = image.copy()
        val_img.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    else:
        val_img = image

    logger.info(f"[PREDICT] stage=validation_downsample_complete | val_dimensions={val_img.size}")
    sys.stdout.flush()

    img_np = np.array(val_img.convert("RGB"))
    h, w, _ = img_np.shape

    if h < 64 or w < 64:
        return False, "Image resolution is too low. Please upload a clear fundus image."

    mean_intensity = float(np.mean(img_np))
    if mean_intensity < 10.0:
        return False, "Uploaded image is too dark or empty. Please upload an illuminated fundus retina image."
    if mean_intensity > 240.0:
        return False, "Uploaded image is overexposed. Please upload a clear fundus retina image."

    r_mean = float(np.mean(img_np[:, :, 0]))
    g_mean = float(np.mean(img_np[:, :, 1]))
    b_mean = float(np.mean(img_np[:, :, 2]))

    if r_mean < b_mean * 1.05 and r_mean < 40.0:
        return False, "The uploaded image does not appear to be a retinal fundus image. Please upload a valid retina scan."

    center_h_start, center_h_end = int(h * 0.2), int(h * 0.8)
    center_w_start, center_w_end = int(w * 0.2), int(w * 0.8)
    center_crop = img_np[center_h_start:center_h_end, center_w_start:center_w_end]

    center_r = float(np.mean(center_crop[:, :, 0]))
    center_b = float(np.mean(center_crop[:, :, 2]))

    if center_r < center_b * 1.08 and center_r < 35.0:
        return False, "The image does not match the color profile of a retinal fundus scan. Please upload a valid retina image."

    logger.info("[PREDICT] stage=validation_blur_check_start")
    sys.stdout.flush()
    gray = np.mean(img_np, axis=2).astype(np.float32)
    if gray.shape[0] > 10 and gray.shape[1] > 10:
        laplacian = (
            gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:] + gray[1:-1, :-2] - 4 * gray[1:-1, 1:-1]
        )
        blur_variance = float(np.var(laplacian))
        logger.info(f"[PREDICT] stage=validation_blur_check_complete | blur_variance={blur_variance:.2f}")
        sys.stdout.flush()
        if blur_variance < 5.0:
            return False, "Image is too blurry or out of focus to perform diabetic retinopathy screening. Please upload a clearer capture."

    return True, ""


def generate_gradcam(
    target_model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
) -> tuple[str, float]:
    """Computes Gradient-Weighted Class Activation Mapping (Grad-CAM) for target class."""
    cam_start = time.perf_counter()
    if hasattr(target_model, "head_conv"):
        target_layer = target_model.head_conv.conv
    elif hasattr(target_model, "features"):
        target_layer = target_model.features[-1]
    else:
        target_layer = list(target_model.children())[-2]

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        if grad_out and len(grad_out) > 0 and grad_out[0] is not None:
            gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        with torch.enable_grad():
            input_var = image_tensor.clone().detach().requires_grad_(True)
            target_model.zero_grad()
            logits = target_model(input_var)
            score = logits[0, target_class]
            score.backward()

        if not activations or not gradients:
            logger.warning("[GRAD-CAM] No activations or gradients captured.")
            return "", round((time.perf_counter() - cam_start) * 1000, 2)

        act = activations[0]
        grad = gradients[0]

        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(image_tensor.shape[2], image_tensor.shape[3]), mode="bilinear", align_corners=False)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        cam_np = (cam.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)

        h_dim, w_dim = cam_np.shape
        rgba = np.zeros((h_dim, w_dim, 4), dtype=np.uint8)
        rgba[:, :, 0] = np.clip(cam_np * 2.0, 0, 255).astype(np.uint8)
        rgba[:, :, 1] = np.clip(255 - np.abs(cam_np.astype(np.float32) - 128) * 2, 0, 255).astype(np.uint8)
        rgba[:, :, 2] = np.clip(255 - cam_np.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
        rgba[:, :, 3] = np.clip(cam_np.astype(np.float32) * 0.75, 0, 200).astype(np.uint8)

        overlay_img = Image.fromarray(rgba, mode="RGBA")
        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        cam_ms = round((time.perf_counter() - cam_start) * 1000, 2)
        return f"data:image/png;base64,{b64_str}", cam_ms
    except Exception as exc:
        logger.warning(f"[GRAD-CAM] Grad-CAM generation failed: {exc}")
        logger.exception(exc)
        cam_ms = round((time.perf_counter() - cam_start) * 1000, 2)
        return "", cam_ms
    finally:
        h1.remove()
        h2.remove()
        target_model.zero_grad()


def load_model() -> None:
    global model, class_names, transform, active_checkpoint_path, loaded_experiment_id

    from backend.netv2_reconstruction import MATLAB_FULL_WEIGHTS_PATH, MatlabNetV2Reconstructed

    if MATLAB_FULL_WEIGHTS_PATH.is_file():
        logger.info(f"Loading 100% reconstructed MATLAB netV2 model from: {MATLAB_FULL_WEIGHTS_PATH}")
        rec_model = MatlabNetV2Reconstructed(MATLAB_FULL_WEIGHTS_PATH)
        model = rec_model.to(DEVICE).eval()
        class_names = ["0", "1", "2", "3", "4"]
        active_checkpoint_path = str(MATLAB_FULL_WEIGHTS_PATH)
        loaded_experiment_id = MODEL_VERSION
        transform = None
    else:
        target_path = E02_CHECKPOINT_PATH if E02_CHECKPOINT_PATH.is_file() else BASELINE_PATH
        if not target_path.is_file():
            raise FileNotFoundError(f"No checkpoint or weights file found at: {MATLAB_FULL_WEIGHTS_PATH}")

        logger.info(f"Loading legacy PyTorch model from: {target_path}")
        checkpoint = torch.load(target_path, map_location=DEVICE, weights_only=False)
        class_names = checkpoint.get("class_names", ["0", "1", "2", "3", "4"])
        loaded_model = efficientnet_b0(weights=None)
        loaded_model.classifier[1] = torch.nn.Sequential(
            torch.nn.Dropout(p=0.30),
            torch.nn.Linear(loaded_model.classifier[1].in_features, len(class_names)),
        )
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        model = loaded_model.to(DEVICE).eval()
        active_checkpoint_path = str(target_path)
        loaded_experiment_id = checkpoint.get("exp_id", "baseline")
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((384, 384), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


@app.on_event("startup")
def startup() -> None:
    load_model()
    from backend.netv2_reconstruction import MatlabNetV2Reconstructed

    rss_mb = get_process_rss_mb()
    logger.info("================ DRISHTIMITRA BACKEND STARTUP DIAGNOSTICS ================")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Model version: {MODEL_VERSION}")
    logger.info(f"Model instance: {type(model).__name__ if model else 'None'}")
    logger.info(f"Device: {DEVICE}")
    if model is not None:
        model_device = next(model.parameters()).device.type
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model is on CPU: {model_device == 'cpu'}")
        logger.info(f"Model is in eval mode: {not model.training}")
        logger.info(f"Model parameter count: {param_count:,}")
        if isinstance(model, MatlabNetV2Reconstructed):
            logger.info("Expected input shape: [1, 3, 224, 224]")
        else:
            logger.info("Expected input shape: [1, 3, 384, 384]")
    logger.info(f"Active checkpoint path: {active_checkpoint_path}")
    logger.info(f"Checkpoint file exists: {Path(active_checkpoint_path).is_file() if active_checkpoint_path else False}")
    logger.info(f"Process RSS memory at startup: {rss_mb} MB")
    logger.info("==========================================================================")
    sys.stdout.flush()


@app.get("/health")
def health() -> dict[str, object]:
    rss_mb = get_process_rss_mb()
    return {
        "status": "ready" if model is not None else "model_not_trained",
        "model_version": MODEL_VERSION,
        "model_description": "Full PyTorch reconstruction of MATLAB-trained EfficientNet-B0 V2 (100% weights & BatchNorm stats loaded)",
        "device": str(DEVICE),
        "loaded_checkpoint": active_checkpoint_path,
        "experiment_id": loaded_experiment_id,
        "target_layer": "head_conv.conv (Reconstructed EfficientNet-B0 final conv layer)",
        "process_rss_mb": rss_mb,
        "medical_disclaimer": "AI attention heatmap generated via PyTorch Grad-CAM (auxiliary explainability representation). Experimental software only; not for clinical diagnosis.",
    }


@app.post("/debug/inference")
async def debug_inference(image: UploadFile = File(...)) -> dict[str, object]:
    """Temporary diagnostic endpoint executing ONLY: decode -> preprocess -> tensor -> model forward pass (NO Grad-CAM)."""
    start_time = time.perf_counter()
    rss_start = get_process_rss_mb()
    logger.info(f"[DEBUG_INFERENCE] request received | filename={image.filename} | rss_mb={rss_start}MB")
    sys.stdout.flush()

    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    contents = await image.read(MAX_IMAGE_BYTES + 1)
    if len(contents) == 0 or len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Invalid file size.")

    with Image.open(io.BytesIO(contents)) as uploaded:
        rgb_image = uploaded.convert("RGB")

    from backend.netv2_reconstruction import MatlabNetV2Reconstructed

    if isinstance(model, MatlabNetV2Reconstructed):
        resized_image = rgb_image.resize((224, 224), resample=Image.Resampling.BILINEAR)
        image_tensor = model.preprocess_image(np.array(resized_image)).to(DEVICE)
    else:
        padded_image = pad_to_square(rgb_image)
        image_tensor = transform(padded_image).unsqueeze(0).to(DEVICE)

    rss_before_inf = get_process_rss_mb()
    logger.info(f"[DEBUG_INFERENCE] before model forward pass | tensor_shape={list(image_tensor.shape)} | dtype={image_tensor.dtype} | rss_mb={rss_before_inf}MB")
    sys.stdout.flush()

    inf_start = time.perf_counter()
    with torch.inference_mode():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    inf_ms = round((time.perf_counter() - inf_start) * 1000, 2)
    grade = int(max(range(len(probs)), key=probs.__getitem__))
    rss_after_inf = get_process_rss_mb()

    logger.info(f"[DEBUG_INFERENCE] model forward pass complete | grade={grade} | confidence={probs[grade]:.4f} | latency={inf_ms}ms | rss_mb={rss_after_inf}MB")
    sys.stdout.flush()

    return {
        "status": "ok",
        "predicted_grade": grade,
        "confidence": round(probs[grade], 4),
        "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
        "input_shape": list(image_tensor.shape),
        "input_dtype": str(image_tensor.dtype),
        "device": str(DEVICE),
        "inference_latency_ms": inf_ms,
        "total_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "rss_before_inf_mb": rss_before_inf,
        "rss_after_inf_mb": rss_after_inf,
        "model_class": type(model).__name__,
    }


@app.post("/debug/gradcam")
async def debug_gradcam(image: UploadFile = File(...)) -> dict[str, object]:
    """Temporary diagnostic endpoint executing ONLY: decode -> preprocess -> tensor -> model forward pass -> Grad-CAM."""
    start_time = time.perf_counter()
    rss_start = get_process_rss_mb()
    logger.info(f"[DEBUG_GRADCAM] request received | filename={image.filename} | rss_mb={rss_start}MB")
    sys.stdout.flush()

    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    contents = await image.read(MAX_IMAGE_BYTES + 1)
    if len(contents) == 0 or len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Invalid file size.")

    with Image.open(io.BytesIO(contents)) as uploaded:
        rgb_image = uploaded.convert("RGB")

    from backend.netv2_reconstruction import MatlabNetV2Reconstructed

    if isinstance(model, MatlabNetV2Reconstructed):
        resized_image = rgb_image.resize((224, 224), resample=Image.Resampling.BILINEAR)
        image_tensor = model.preprocess_image(np.array(resized_image)).to(DEVICE)
    else:
        padded_image = pad_to_square(rgb_image)
        image_tensor = transform(padded_image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    grade = int(max(range(len(probs)), key=probs.__getitem__))

    rss_before_cam = get_process_rss_mb()
    logger.info(f"[DEBUG_GRADCAM] before gradcam | grade={grade} | rss_mb={rss_before_cam}MB")
    sys.stdout.flush()

    cam_b64, cam_ms = generate_gradcam(model, image_tensor, grade)
    rss_after_cam = get_process_rss_mb()

    logger.info(f"[DEBUG_GRADCAM] gradcam complete | heatmap_len={len(cam_b64)} | latency={cam_ms}ms | rss_mb={rss_after_cam}MB")
    sys.stdout.flush()

    return {
        "status": "ok",
        "predicted_grade": grade,
        "gradcam_latency_ms": cam_ms,
        "heatmap_len": len(cam_b64),
        "total_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "rss_before_cam_mb": rss_before_cam,
        "rss_after_cam_mb": rss_after_cam,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    predict_start = time.perf_counter()
    stage = "request_received"

    try:
        rss_req = get_process_rss_mb()
        logger.info(f"[PREDICT] stage=request_received | filename={image.filename} | content_type={image.content_type} | rss_mb={rss_req}MB")
        sys.stdout.flush()

        if model is None:
            raise HTTPException(status_code=503, detail="No trained model found.")

        content_type = (image.content_type or "").lower()
        if content_type and not (content_type.startswith("image/") or content_type == "application/octet-stream"):
            logger.warning(f"[PREDICT][WARN] Invalid content_type: {content_type}")
            raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP fundus image.")

        contents = await image.read(MAX_IMAGE_BYTES + 1)
        if len(contents) > MAX_IMAGE_BYTES:
            logger.warning(f"[PREDICT][WARN] Image size exceeds limit: {len(contents)} > {MAX_IMAGE_BYTES}")
            raise HTTPException(status_code=413, detail="Image exceeds the 12 MB limit.")
        if len(contents) == 0:
            logger.warning("[PREDICT][WARN] Empty image file received.")
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        # Stage 1: image_decoded
        stage = "image_decoded"
        try:
            with Image.open(io.BytesIO(contents)) as uploaded:
                rgb_image = uploaded.convert("RGB")
                orig_w, orig_h = rgb_image.size
                if orig_w > 1024 or orig_h > 1024:
                    rgb_image.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
        except (UnidentifiedImageError, OSError) as error:
            logger.warning(f"[PREDICT][WARN] Failed to decode image: {error}")
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error

        del contents
        rss_resized = get_process_rss_mb()
        logger.info(f"[PREDICT] stage=image_resized | orig_dimensions={orig_w}x{orig_h} | working_dimensions={rgb_image.size} | rss_mb={rss_resized}MB")
        sys.stdout.flush()

        # Stage 2: validation_start / validation_complete
        stage = "validation_start"
        logger.info("[PREDICT] stage=validation_start")
        sys.stdout.flush()
        
        is_valid_retina, error_reason = validate_retina_image(rgb_image)
        if not is_valid_retina:
            logger.warning(f"[PREDICT][WARN] Retina validation rejected: {error_reason}")
            raise HTTPException(status_code=400, detail=error_reason)

        stage = "validation_complete"
        rss_val = get_process_rss_mb()
        logger.info(f"[PREDICT] stage=validation_complete | rss_mb={rss_val}MB")
        sys.stdout.flush()

        from backend.netv2_reconstruction import MatlabNetV2Reconstructed

        # Stage 3: preprocess_start / preprocess_complete
        stage = "preprocess_start"
        rss_prep_start = get_process_rss_mb()
        logger.info(f"[PREDICT] stage=preprocess_start | rss_mb={rss_prep_start}MB")
        sys.stdout.flush()

        if isinstance(model, MatlabNetV2Reconstructed):
            logger.info("[PREDICT] stage=pil_resize_start | target_size=(224, 224)")
            sys.stdout.flush()
            resized_image = rgb_image.resize((224, 224), resample=Image.Resampling.BILINEAR)
            logger.info("[PREDICT] stage=pil_resize_complete")
            sys.stdout.flush()

            logger.info("[PREDICT] stage=numpy_convert_start")
            sys.stdout.flush()
            arr_224 = np.array(resized_image)
            arr_mb = arr_224.nbytes / (1024 * 1024)
            logger.info(f"[PREDICT] stage=numpy_convert_complete | array_shape={arr_224.shape} | array_dtype={arr_224.dtype} | memory_size={arr_mb:.3f}MB")
            sys.stdout.flush()

            logger.info("[PREDICT] stage=tensor_conversion_start")
            sys.stdout.flush()
            image_tensor = model.preprocess_image(arr_224).to(DEVICE)
            rss_tensor = get_process_rss_mb()
            logger.info(f"[PREDICT] stage=tensor_conversion_complete | tensor_shape={list(image_tensor.shape)} | dtype={image_tensor.dtype} | device={image_tensor.device} | rss_mb={rss_tensor}MB")
            sys.stdout.flush()
        else:
            padded_image = pad_to_square(rgb_image)
            image_tensor = transform(padded_image).unsqueeze(0).to(DEVICE)

        stage = "preprocess_complete"
        rss_prep_complete = get_process_rss_mb()
        logger.info(f"[PREDICT] stage=preprocess_complete | rss_mb={rss_prep_complete}MB")
        sys.stdout.flush()

        # Stage 4: model_inference_start / model_inference_complete
        stage = "model_inference_start"
        diag = get_model_diagnostics(model, image_tensor)
        rss_before_inf = get_process_rss_mb()
        logger.info(f"[PREDICT] stage=before_model_inference | diagnostics={diag} | rss_mb={rss_before_inf}MB")
        sys.stdout.flush()

        inf_start = time.perf_counter()
        with torch.inference_mode():
            p_orig = torch.softmax(model(image_tensor), dim=1)
            if isinstance(model, MatlabNetV2Reconstructed):
                probs_tensor = p_orig
            else:
                p_hflip = torch.softmax(model(torch.flip(image_tensor, dims=[3])), dim=1)
                probs_tensor = (p_orig + p_hflip) / 2.0
            probabilities = probs_tensor[0].cpu().tolist()
        inf_ms = round((time.perf_counter() - inf_start) * 1000, 2)
        grade = int(max(range(len(probabilities)), key=probabilities.__getitem__))

        stage = "model_inference_complete"
        rss_after_inf = get_process_rss_mb()
        logger.info(f"[PREDICT] stage=after_model_inference | grade={grade} | confidence={probabilities[grade]:.4f} | latency={inf_ms}ms | rss_mb={rss_after_inf}MB")
        sys.stdout.flush()

        # Stage 5: gradcam_start / gradcam_complete
        stage = "gradcam_start"
        rss_before_cam = get_process_rss_mb()
        logger.info(f"[PREDICT] stage=before_gradcam | rss_mb={rss_before_cam}MB")
        sys.stdout.flush()

        heatmap_b64, gradcam_ms = generate_gradcam(model, image_tensor, grade)

        stage = "gradcam_complete"
        rss_after_cam = get_process_rss_mb()
        logger.info(f"[PREDICT] stage=after_gradcam | heatmap_len={len(heatmap_b64)} | latency={gradcam_ms}ms | rss_mb={rss_after_cam}MB")
        sys.stdout.flush()

        # Stage 6: response_build_start / response_build_complete
        stage = "response_build_start"
        logger.info("[PREDICT] stage=response_build_start")
        sys.stdout.flush()

        total_ms = round((time.perf_counter() - predict_start) * 1000, 2)
        response_data = {
            "model_version": MODEL_VERSION,
            "grade": grade,
            "label": class_names[grade] if grade < len(class_names) else str(grade),
            "confidence": round(probabilities[grade], 4),
            "probabilities": {str(index): round(probability, 4) for index, probability in enumerate(probabilities)},
            "heatmap": heatmap_b64,
            "medical_disclaimer": "AI attention heatmap highlighting regions that influenced model prediction. Experimental software only; not for clinical diagnosis.",
            "performance_metrics": {
                "prediction_latency_ms": total_ms,
                "gradcam_latency_ms": gradcam_ms,
                "inference_strategy": "Direct Forward Pass" if isinstance(model, MatlabNetV2Reconstructed) else "AVG_2X (Original + Horizontal Flip TTA Average)",
                "forward_passes": 1 if isinstance(model, MatlabNetV2Reconstructed) else 2,
                "experiment_id": loaded_experiment_id,
            },
        }

        stage = "response_build_complete"
        logger.info(f"[PREDICT] stage=response_build_complete | total_latency={total_ms}ms")
        sys.stdout.flush()
        return response_data

    except HTTPException:
        sys.stdout.flush()
        raise
    except Exception as exc:
        logger.error(f"[PREDICT][ERROR] stage={stage}")
        logger.exception(f"[PREDICT][EXCEPTION] Unhandled error at stage '{stage}': {exc}")
        sys.stdout.flush()
        raise HTTPException(
            status_code=502,
            detail=f"Inference failed during stage '{stage}': {type(exc).__name__} - {str(exc)}"
        ) from exc


# Mount compiled React assets and SPA fallback
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path in {"health", "predict", "debug/inference", "debug/gradcam"}:
        raise HTTPException(status_code=404, detail="Not found")
    target_file = DIST_DIR / full_path
    if target_file.is_file():
        return FileResponse(target_file)
    index_file = DIST_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {"message": "DrishtiMitra Inference API active. Run 'npm run build' to serve frontend."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
