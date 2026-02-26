import numpy as np
import io
import os
import cv2
import json
import base64

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.http import HttpResponse
from django.conf import settings
from PIL import Image

from drf_spectacular.utils import extend_schema, OpenApiTypes

from .views_cnn_vit import MODEL_PATHS, CLASSES, load_pytorch_model, LOADED_MODELS


def _get_model(model_name: str):
    """Încarcă modelul ResNet din cache sau de pe disc."""
    if model_name not in LOADED_MODELS:
        path_candidate = MODEL_PATHS[model_name]
        if not os.path.exists(path_candidate):
            base_parent = os.path.dirname(os.path.dirname(settings.BASE_DIR))
            full_path = os.path.join(base_parent, path_candidate)
            if not os.path.exists(full_path):
                full_path = os.path.join(
                    settings.BASE_DIR, "Models", "CNN",
                    os.path.basename(path_candidate)
                )
        else:
            full_path = path_candidate
        LOADED_MODELS[model_name] = load_pytorch_model(full_path, model_name)
    return LOADED_MODELS[model_name]


def _predict_fn(images_np: np.ndarray, model, device) -> np.ndarray:
    """
    Funcție de predicție pentru LIME.
    images_np: array (N, H, W, 3) uint8
    Returnează array (N, num_classes) probabilități
    """
    import torch
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval()
    results = []
    with torch.no_grad():
        for img_np in images_np:
            pil_img = Image.fromarray(img_np.astype(np.uint8))
            tensor = preprocess(pil_img).unsqueeze(0).to(device)
            output = model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
            results.append(probs)
    return np.array(results)


def generate_lime_segments(image_pil: Image.Image):
    """
    Generează segmentele SLIC pentru imaginea dată.
    Returnează array de segmente.
    """
    from skimage.segmentation import slic

    img_np = np.array(image_pil.convert('RGB'))
    segments = slic(img_np, n_segments=50, compactness=10, sigma=1, start_label=0)
    return segments, img_np


def apply_mask(image_np: np.ndarray, segments: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Aplică o mască pe segmente. Segmentele cu mask=0 devin gri (128).
    """
    masked = image_np.copy()
    for seg_id in range(mask.shape[0]):
        if mask[seg_id] == 0:
            masked[segments == seg_id] = 0  # negru = segment dezactivat
    return masked


def run_lime(image_pil: Image.Image, model, device, n_samples: int = 200, n_perturbations_display: int = 3):
    """
    Rulează LIME pe imagine.
    Returnează:
      - lime_heatmap: Image PIL cu heatmap
      - perturbed_images: listă de Image PIL (perturbate)
      - predicted_class: str
    """
    from skimage.segmentation import mark_boundaries
    import torch

    img_np = np.array(image_pil.convert('RGB'))
    orig_h, orig_w = img_np.shape[:2]

    segments, _ = generate_lime_segments(image_pil)
    n_segments = segments.max() + 1

    # ---- Predicție pe imaginea originală ----
    probs_orig = _predict_fn(np.array([img_np]), model, device)[0]
    predicted_class_idx = int(np.argmax(probs_orig))
    predicted_class = CLASSES[predicted_class_idx]

    # ---- Generăm perturbări aleatoare ----
    rng = np.random.default_rng(42)
    masks = rng.integers(0, 2, size=(n_samples, n_segments))  # (N, S) binar

    perturbed_imgs = []
    for mask in masks:
        perturbed = apply_mask(img_np, segments, mask)
        perturbed_imgs.append(perturbed)

    perturbed_imgs_np = np.array(perturbed_imgs)  # (N, H, W, 3)

    # ---- Predicții pe perturbări ----
    all_probs = _predict_fn(perturbed_imgs_np, model, device)  # (N, num_classes)
    labels = all_probs[:, predicted_class_idx]  # (N,)

    # ---- Fit model liniar simplu (Ridge Regression) ----
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(masks, labels)
    coef = ridge.coef_  # (n_segments,)

    # ---- Construim heatmap LIME ----
    heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
    for seg_id in range(n_segments):
        heatmap[segments == seg_id] = coef[seg_id]

    # Normalizare separată pentru pozitive/negative
    pos_mask = heatmap > 0
    neg_mask = heatmap < 0

    heatmap_vis = np.zeros_like(heatmap)
    if pos_mask.any():
        heatmap_vis[pos_mask] = heatmap[pos_mask] / heatmap[pos_mask].max()
    if neg_mask.any():
        heatmap_vis[neg_mask] = heatmap[neg_mask] / abs(heatmap[neg_mask].min())

    # Colormap: verde = important pozitiv, roșu = important negativ
    heatmap_color = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    # Pozitiv → verde
    pos_intensity = np.clip(heatmap_vis, 0, 1)
    heatmap_color[..., 1] = (pos_intensity * 255).astype(np.uint8)
    # Negativ → roșu
    neg_intensity = np.clip(-heatmap_vis, 0, 1)
    heatmap_color[..., 0] = (neg_intensity * 255).astype(np.uint8)

    # Suprapunem pe imaginea originală
    blended = cv2.addWeighted(img_np, 0.55, heatmap_color, 0.45, 0)
    # Desenăm granițele segmentelor
    blended_with_borders = mark_boundaries(blended, segments, color=(1, 1, 1), mode='thin')
    blended_pil = Image.fromarray((blended_with_borders * 255).astype(np.uint8))

    # ---- Selectăm 3 perturbări reprezentative ----
    # Sortăm perturbările după cât de diferit este label față de original:
    # - o perturbate cu predicted_class_idx scăzut (mult mascat)
    # - o perturbate medie
    # - perturbarea cu cel mai mare score (similar cu originalul)
    sorted_idx = np.argsort(labels)
    low_idx = sorted_idx[0]
    mid_idx = sorted_idx[len(sorted_idx) // 2]
    high_idx = sorted_idx[-1]

    selected_perturbed = []
    selected_labels = []
    for idx in [low_idx, mid_idx, high_idx]:
        pil_img = Image.fromarray(perturbed_imgs_np[idx].astype(np.uint8))
        selected_perturbed.append(pil_img)
        selected_labels.append(float(labels[idx]))

    return blended_pil, selected_perturbed, selected_labels, predicted_class


class LimeView(APIView):
    """
    Endpoint LIME: primește o imagine MRI, rulează LIME cu ResNet
    și returnează un JSON cu:
      - lime_heatmap: imagine PNG în base64
      - perturbed_images: lista de 3 imagini PNG în base64
      - perturbed_scores: lista de 3 scoruri (probabilitatea clasei prezise)
      - predicted_class: clasa prezisă
    """
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        summary="LIME pe imagine (ResNet)",
        description=(
            "Primește o imagine MRI și returnează un JSON cu heatmap-ul LIME "
            "și 3 imagini perturbate reprezentative în base64."
        ),
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'image': {
                        'type': 'string',
                        'format': 'binary',
                        'description': 'Imaginea MRI'
                    }
                },
                'required': ['image']
            }
        },
        responses={
            200: OpenApiTypes.OBJECT,
            400: OpenApiTypes.OBJECT,
            500: OpenApiTypes.OBJECT
        }
    )
    def post(self, request):
        import torch

        if 'image' not in request.FILES:
            return Response({"error": "No image provided"}, status=400)

        image_file = request.FILES['image']
        try:
            raw_image = Image.open(image_file)

            # Dacă are canal alpha → crop la zona non-transparentă
            if raw_image.mode in ('RGBA', 'LA') or (raw_image.mode == 'P' and 'transparency' in raw_image.info):
                raw_rgba = raw_image.convert('RGBA')
                alpha_channel = np.array(raw_rgba.split()[-1])
                rows = np.any(alpha_channel > 10, axis=1)
                cols = np.any(alpha_channel > 10, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    pad = 5
                    rmin = max(0, rmin - pad)
                    rmax = min(alpha_channel.shape[0], rmax + pad)
                    cmin = max(0, cmin - pad)
                    cmax = min(alpha_channel.shape[1], cmax + pad)
                    raw_image = raw_rgba.crop((cmin, rmin, cmax, rmax))

            image_pil = raw_image.convert('RGB')

        except Exception as e:
            return Response({"error": f"Invalid image: {e}"}, status=400)

        model_name = 'resnet101'
        try:
            model = _get_model(model_name)
            device = next(model.parameters()).device
        except Exception as e:
            return Response({"error": f"Failed to load model: {e}"}, status=500)

        try:
            lime_heatmap_pil, perturbed_pils, perturbed_scores, predicted_class = run_lime(
                image_pil, model, device, n_samples=150
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": f"LIME failed: {str(e)}"}, status=500)

        def pil_to_b64(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')

        response_data = {
            "predicted_class": predicted_class,
            "lime_heatmap": pil_to_b64(lime_heatmap_pil),
            "perturbed_images": [pil_to_b64(p) for p in perturbed_pils],
            "perturbed_scores": perturbed_scores,
        }

        return Response(response_data)
