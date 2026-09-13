"""Local FastAPI inference service for the trained DrishtiMitra research model."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision.models import efficientnet_b0
from torchvision.transforms import v2


import time

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
E02_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E02_no_sampler_35ep" / "checkpoint.pt"
BASELINE_PATH = PROJECT_ROOT / "artifacts" / "drishtimitra_efficientnet_b0.pt"
DIST_DIR = PROJECT_ROOT / "dist"
ASSETS_DIR = DIST_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_BYTES = 12 * 1024 * 1024

app = FastAPI(title="DrishtiMitra experimental inference API", version="0.1.0")
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
class_names: list[str] = []
transform: v2.Compose | None = None
active_checkpoint_path: str = ""
loaded_experiment_id: str = ""


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
    img_np = np.array(image.convert("RGB"))
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
    gray = np.mean(img_np, axis=2).astype(np.float32)
    if gray.shape[0] > 10 and gray.shape[1] > 10:
        laplacian = (
            gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:] + gray[1:-1, :-2] - 4 * gray[1:-1, 1:-1]
        )
        blur_variance = float(np.var(laplacian))
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
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        input_var = image_tensor.clone().detach().requires_grad_(True)
        target_model.zero_grad()
        logits = target_model(input_var)
        score = logits[0, target_class]
        score.backward()

        if not activations or not gradients:
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
        print(f"Warning: Grad-CAM generation failed: {exc}")
        cam_ms = round((time.perf_counter() - cam_start) * 1000, 2)
        return "", cam_ms
    finally:
        h1.remove()
        h2.remove()

import scipy.io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATLAB_WEIGHTS_PATH = PROJECT_ROOT / "artifacts" / "matlab_dr_classifier_v2_weights.mat"
E02_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E02_no_sampler_35ep" / "checkpoint.pt"
BASELINE_PATH = PROJECT_ROOT / "artifacts" / "drishtimitra_efficientnet_b0.pt"
DIST_DIR = PROJECT_ROOT / "dist"
ASSETS_DIR = DIST_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "v0.2-matlab-netv2-reconstructed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_BYTES = 12 * 1024 * 1024

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


def load_model() -> None:
    global model, class_names, transform, active_checkpoint_path, loaded_experiment_id

    from backend.netv2_reconstruction import MATLAB_FULL_WEIGHTS_PATH, MatlabNetV2Reconstructed

    if MATLAB_FULL_WEIGHTS_PATH.is_file():
        print(f"Loading 100% reconstructed MATLAB netV2 model from: {MATLAB_FULL_WEIGHTS_PATH}")
        rec_model = MatlabNetV2Reconstructed(MATLAB_FULL_WEIGHTS_PATH)
        model = rec_model.to(DEVICE).eval()
        class_names = ["0", "1", "2", "3", "4"]
        active_checkpoint_path = str(MATLAB_FULL_WEIGHTS_PATH)
        loaded_experiment_id = MODEL_VERSION
        transform = None  # Using custom MatlabNetV2Reconstructed.preprocess_image
    else:
        target_path = E02_CHECKPOINT_PATH if E02_CHECKPOINT_PATH.is_file() else BASELINE_PATH
        if not target_path.is_file():
            raise FileNotFoundError(f"No checkpoint or weights file found at: {MATLAB_FULL_WEIGHTS_PATH}")

        print(f"Loading legacy PyTorch model from: {target_path}")
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
        "medical_disclaimer": "AI attention heatmap generated via PyTorch Grad-CAM (auxiliary explainability representation). Experimental software only; not for clinical diagnosis.",
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    predict_start = time.perf_counter()
    if model is None:
        raise HTTPException(status_code=503, detail="No trained model found.")

    content_type = (image.content_type or "").lower()
    if content_type and not (content_type.startswith("image/") or content_type == "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP fundus image.")

    contents = await image.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB limit.")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(contents)) as uploaded:
            rgb_image = uploaded.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error

    # 1. Quality & Non-retina validation check
    is_valid_retina, error_reason = validate_retina_image(rgb_image)
    if not is_valid_retina:
        raise HTTPException(status_code=400, detail=error_reason)

    from backend.netv2_reconstruction import MatlabNetV2Reconstructed

    # 2. Image Preprocessing & Tensor Conversion
    if isinstance(model, MatlabNetV2Reconstructed):
        # Resize to MATLAB 224x224 input size
        resized_image = rgb_image.resize((224, 224), resample=Image.Resampling.BILINEAR)
        image_tensor = model.preprocess_image(np.array(resized_image)).to(DEVICE)
    else:
        padded_image = pad_to_square(rgb_image)
        image_tensor = transform(padded_image).unsqueeze(0).to(DEVICE)

    # 3. Model Forward Pass
    with torch.no_grad():
        p_orig = torch.softmax(model(image_tensor), dim=1)
        if isinstance(model, MatlabNetV2Reconstructed):
            probs_tensor = p_orig
        else:
            p_hflip = torch.softmax(model(torch.flip(image_tensor, dims=[3])), dim=1)
            probs_tensor = (p_orig + p_hflip) / 2.0
        probabilities = probs_tensor[0].cpu().tolist()
    grade = int(max(range(len(probabilities)), key=probabilities.__getitem__))

    # 4. Grad-CAM Explainability Heatmap & Latency Measurement
    heatmap_b64, gradcam_ms = generate_gradcam(model, image_tensor, grade)
    total_ms = round((time.perf_counter() - predict_start) * 1000, 2)

    return {
        "model_version": MODEL_VERSION,
        "grade": grade,
        "label": class_names[grade],
        "confidence": round(probabilities[grade], 4),
        "probabilities": {str(index): round(probability, 4) for index, probability in enumerate(probabilities)},
        "heatmap": heatmap_b64,
        "medical_disclaimer": "AI attention heatmap highlighting regions that influenced model prediction. Experimental software only; not for clinical diagnosis.",
        "performance_metrics": {
            "prediction_latency_ms": total_ms,
            "gradcam_latency_ms": gradcam_ms,
            "inference_strategy": "AVG_2X (Original + Horizontal Flip TTA Average)",
            "forward_passes": 2,
            "experiment_id": loaded_experiment_id,
        },
    }


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




