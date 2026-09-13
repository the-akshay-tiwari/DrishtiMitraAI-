"""FastAPI inference service for the trained DrishtiMitra research model."""

from __future__ import annotations

import base64
import io
import logging
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets top-level BEFORE route definitions so /assets/* is handled by StaticFiles
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

model: torch.nn.Module | None = None
class_names: list[str] = ["0", "1", "2", "3", "4"]
transform: v2.Compose | None = None
active_checkpoint_path: str = ""
loaded_experiment_id: str = ""


def get_memory_usage_mb() -> float:
    """Returns current traced memory usage in MB using tracemalloc if active."""
    if tracemalloc.is_tracing():
        current, _ = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)
    return 0.0


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
    logger.info(f"[PREDICT] stage=validation_downsample_start | orig_dimensions={w}x{h}")
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

    # In fundus retina images, Red channel dominates over Blue channel (R > B)
    if r_mean < b_mean * 1.05 and r_mean < 40.0:
        return False, "The uploaded image does not appear to be a retinal fundus image. Please upload a valid retina scan."

    # Center crop check (retina circular field)
    center_h_start, center_h_end = int(h * 0.2), int(h * 0.8)
    center_w_start, center_w_end = int(w * 0.2), int(w * 0.8)
    center_crop = img_np[center_h_start:center_h_end, center_w_start:center_w_end]

    center_r = float(np.mean(center_crop[:, :, 0]))
    center_b = float(np.mean(center_crop[:, :, 2]))

    if center_r < center_b * 1.08 and center_r < 35.0:
        return False, "The image does not match the color profile of a retinal fundus scan. Please upload a valid retina image."

    # Blur / Quality Check using discrete Laplacian variance
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
    """Computes Gradient-Weighted Class Activation Mapping (Grad-CAM) for target class.
    
    Generates an AI attention/explainability heatmap showing regions influencing prediction.
    It does NOT segment or identify specific retinal lesions.
    """
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
    tracemalloc.start()
    load_model()
    from backend.netv2_reconstruction import MatlabNetV2Reconstructed

    logger.info("================ DRISHTIMITRA BACKEND STARTUP DIAGNOSTICS ================")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Model version: {MODEL_VERSION}")
    logger.info(f"Model instance: {type(model).__name__ if model else 'None'}")
    logger.info(f"Device: {DEVICE}")
    if model is not None:
        model_device = next(model.parameters()).device.type
        logger.info(f"Model is on CPU: {model_device == 'cpu'}")
        logger.info(f"Model is in eval mode: {not model.training}")
        if isinstance(model, MatlabNetV2Reconstructed):
            logger.info("Expected input shape: [1, 3, 224, 224]")
        else:
            logger.info("Expected input shape: [1, 3, 384, 384]")
    logger.info(f"Active checkpoint path: {active_checkpoint_path}")
    logger.info(f"Checkpoint file exists: {Path(active_checkpoint_path).is_file() if active_checkpoint_path else False}")
    logger.info(f"Startup memory tracing active: {tracemalloc.is_tracing()}")
    logger.info("==========================================================================")
    sys.stdout.flush()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ready" if model is not None else "model_not_trained",
        "model_version": MODEL_VERSION,
        "model_description": "Full PyTorch reconstruction of MATLAB-trained EfficientNet-B0 V2 (100% weights & BatchNorm stats loaded)",
        "device": str(DEVICE),
        "loaded_checkpoint": active_checkpoint_path,
        "experiment_id": loaded_experiment_id,
        "target_layer": "head_conv.conv (Reconstructed EfficientNet-B0 final conv layer)",
        "traced_memory_mb": round(get_memory_usage_mb(), 2),
        "medical_disclaimer": "AI attention heatmap generated via PyTorch Grad-CAM (auxiliary explainability representation). Experimental software only; not for clinical diagnosis.",
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    predict_start = time.perf_counter()
    stage = "request_received"

    try:
        logger.info(f"[PREDICT] stage=request_received | filename={image.filename} | content_type={image.content_type}")
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
        except (UnidentifiedImageError, OSError) as error:
            logger.warning(f"[PREDICT][WARN] Failed to decode image: {error}")
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error

        orig_w, orig_h = rgb_image.size
        logger.info(f"[PREDICT] stage=image_decoded | mode={rgb_image.mode} | dimensions={orig_w}x{orig_h} | bytes={len(contents)}")
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
        logger.info("[PREDICT] stage=validation_complete")
        sys.stdout.flush()

        from backend.netv2_reconstruction import MatlabNetV2Reconstructed

        # Stage 3: preprocess_start / preprocess_complete
        stage = "preprocess_start"
        logger.info(f"[PREDICT] stage=preprocess_start | mem_before_mb={get_memory_usage_mb():.2f}MB")
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
            logger.info(f"[PREDICT] stage=tensor_conversion_complete | tensor_shape={list(image_tensor.shape)} | dtype={image_tensor.dtype} | device={image_tensor.device}")
            sys.stdout.flush()
        else:
            padded_image = pad_to_square(rgb_image)
            image_tensor = transform(padded_image).unsqueeze(0).to(DEVICE)

        stage = "preprocess_complete"
        logger.info(f"[PREDICT] stage=preprocess_complete | mem_after_mb={get_memory_usage_mb():.2f}MB")
        sys.stdout.flush()

        # Stage 4: model_inference_start / model_inference_complete
        stage = "model_inference_start"
        logger.info(f"[PREDICT] stage=model_inference_start | model={type(model).__name__}")
        sys.stdout.flush()

        inf_start = time.perf_counter()
        with torch.no_grad():
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
        logger.info(f"[PREDICT] stage=model_inference_complete | grade={grade} | confidence={probabilities[grade]:.4f} | latency={inf_ms}ms")
        sys.stdout.flush()

        # Stage 5: gradcam_start / gradcam_complete
        stage = "gradcam_start"
        logger.info("[PREDICT] stage=gradcam_start")
        sys.stdout.flush()

        heatmap_b64, gradcam_ms = generate_gradcam(model, image_tensor, grade)

        stage = "gradcam_complete"
        logger.info(f"[PREDICT] stage=gradcam_complete | heatmap_len={len(heatmap_b64)} | latency={gradcam_ms}ms")
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
    if full_path in {"health", "predict"}:
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
