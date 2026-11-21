from __future__ import annotations

import base64
import io
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
FRONTEND_DIR = ROOT_DIR / "frontend"
CLASS_NAMES = [
    "Atelectasis",
    "Normal",
    "Pneumonia",
    "Pulmonary Edema",
    "Tuberculosis",
]
MODEL_FILES = {
    "DenseNet121": "DenseNet121_best.pth",
    "EfficientNetV2-S": "EfficientNetV2_best.pth",
    "ResNet50": "ResNet50_best.pth",
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
logger = logging.getLogger("chest_xray")
prediction_lock = threading.Lock()

image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def build_models() -> dict[str, nn.Module]:
    """Create the exact torchvision architectures used by the checkpoints."""
    densenet = models.densenet121(weights=None)
    densenet.classifier = nn.Linear(1024, len(CLASS_NAMES))

    efficientnet = models.efficientnet_v2_s(weights=None)
    efficientnet.classifier[1] = nn.Linear(1280, len(CLASS_NAMES))

    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(2048, len(CLASS_NAMES))

    return {
        "DenseNet121": densenet,
        "EfficientNetV2-S": efficientnet,
        "ResNet50": resnet,
    }


def _checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to read checkpoint {path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint {path} does not contain a state dictionary.")

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Checkpoint {path} contains no model parameters.")

    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    return state_dict


def load_models() -> tuple[dict[str, nn.Module], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_models = build_models()

    for name, model in loaded_models.items():
        checkpoint_path = CHECKPOINT_DIR / MODEL_FILES[name]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Required checkpoint is missing: {checkpoint_path}")

        model.load_state_dict(_checkpoint_state_dict(checkpoint_path), strict=True)
        model.to(device)
        model.eval()
        logger.info("Loaded %s checkpoint from %s", name, checkpoint_path)

    return loaded_models, device


def _probability_result(probabilities: np.ndarray) -> dict[str, Any]:
    class_index = int(np.argmax(probabilities))
    return {
        "prediction": CLASS_NAMES[class_index],
        "confidence": float(probabilities[class_index]),
        "probabilities": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(CLASS_NAMES)
        },
    }


def generate_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    tensor: torch.Tensor,
    class_index: int,
) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def capture_activation(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    hook = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor)
            logits[0, class_index].backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM did not capture activations and gradients.")

        activation = activations[0].detach()
        gradient = gradients[0].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def _jet_colormap(cam: np.ndarray) -> np.ndarray:
    value = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _make_overlay(image: Image.Image, cam: np.ndarray) -> str:
    image_array = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    heatmap = _jet_colormap(cam)
    overlay = np.clip(0.58 * image_array + 0.42 * heatmap, 0.0, 1.0)
    overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))
    return _encode_png(overlay_image)


def analyze_image(
    image: Image.Image,
    loaded_models: dict[str, nn.Module],
    device: torch.device,
    target_layers: dict[str, nn.Module],
) -> dict[str, Any]:
    rgb_image = image.convert("RGB").resize((224, 224))
    tensor = image_transform(rgb_image).unsqueeze(0).to(device)
    model_probabilities: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for name, model in loaded_models.items():
            model_probabilities[name] = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    ensemble_probabilities = np.mean(list(model_probabilities.values()), axis=0)
    ensemble_class = int(np.argmax(ensemble_probabilities))
    model_results = {
        name: _probability_result(probabilities)
        for name, probabilities in model_probabilities.items()
    }

    gradcams = {
        name: generate_gradcam(
            loaded_models[name],
            target_layers[name],
            tensor,
            ensemble_class,
        )
        for name in MODEL_FILES
    }
    ensemble_cam = np.mean(list(gradcams.values()), axis=0)

    visualizations = {
        "original": _encode_png(rgb_image),
        "DenseNet121": _make_overlay(rgb_image, gradcams["DenseNet121"]),
        "EfficientNetV2-S": _make_overlay(rgb_image, gradcams["EfficientNetV2-S"]),
        "ResNet50": _make_overlay(rgb_image, gradcams["ResNet50"]),
        "Ensemble": _make_overlay(rgb_image, ensemble_cam),
    }

    return {
        "ensemble": _probability_result(ensemble_probabilities),
        "individual_models": model_results,
        "visualizations": visualizations,
    }


@asynccontextmanager
async def lifespan(application: FastAPI):
    loaded_models, device = load_models()
    application.state.models = loaded_models
    application.state.device = device
    application.state.target_layers = {
        "DenseNet121": loaded_models["DenseNet121"].features.denseblock4,
        "EfficientNetV2-S": loaded_models["EfficientNetV2-S"].features[-1],
        "ResNet50": loaded_models["ResNet50"].layer4[-1],
    }
    yield


app = FastAPI(
    title="Chest X-Ray AI",
    description="Three-model chest X-ray classification and Grad-CAM visualization.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    loaded_models = getattr(app.state, "models", {})
    return {
        "status": "ok" if len(loaded_models) == len(MODEL_FILES) else "starting",
        "device": str(getattr(app.state, "device", "not loaded")),
        "models": list(loaded_models),
        "classes": CLASS_NAMES,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB upload limit.")

    try:
        with Image.open(io.BytesIO(contents)) as uploaded_image:
            if uploaded_image.width * uploaded_image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions are too large to process.",
                )
            image = uploaded_image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Upload a valid image file.") from exc

    try:
        with prediction_lock:
            return analyze_image(
                image,
                app.state.models,
                app.state.device,
                app.state.target_layers,
            )
    except Exception as exc:
        logger.exception("X-ray analysis failed")
        raise HTTPException(status_code=500, detail="Model analysis failed.") from exc


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
