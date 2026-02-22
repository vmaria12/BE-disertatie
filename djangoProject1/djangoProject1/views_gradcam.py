import numpy as np
import io
import os
import cv2

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import HttpResponse
from django.conf import settings
from PIL import Image

from drf_spectacular.utils import extend_schema, OpenApiTypes

from .views_cnn_vit import MODEL_PATHS, CLASSES, load_pytorch_model, LOADED_MODELS


def generate_gradcam(model, image_pil, target_class_idx=None):
    """
    Generează o hartă Grad-CAM pentru un model ResNet/ViT.
    Returnează imaginea PIL cu heatmap suprapus și clasa prezisă.
    """
    import torch
    from torchvision import transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    input_tensor.requires_grad = False

    # Găsim ultimul layer Conv2d
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise ValueError("Nu s-a găsit niciun layer Conv2d în model.")

    # Storage pentru activări și gradienți
    activation_storage = {}
    gradient_storage = {}

    def save_activation(module, input, output):
        activation_storage['value'] = output.detach()

    def save_gradient(module, grad_input, grad_output):
        # grad_output[0] este gradientul față de ieșirea layer-ului
        gradient_storage['value'] = grad_output[0].detach()

    # Înregistrăm hook-urile
    handle_fwd = target_layer.register_forward_hook(save_activation)
    # register_full_backward_hook are aceeași signatură dar fără warning-ul PyTorch
    try:
        handle_bwd = target_layer.register_full_backward_hook(save_gradient)
    except AttributeError:
        # Fallback pentru versiuni vechi de PyTorch
        handle_bwd = target_layer.register_backward_hook(save_gradient)

    # Forward pass cu gradienți activi
    model.eval()
    output = model(input_tensor)

    # Selectăm clasa țintă (dacă nu e specificată, folosim clasa prezisă)
    if target_class_idx is None:
        target_class_idx = int(output.argmax(dim=1).item())

    # Backward pass
    model.zero_grad()
    score = output[0, target_class_idx]
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    if 'value' not in activation_storage or 'value' not in gradient_storage:
        raise ValueError("Hook-urile nu au capturat activări/gradienți.")

    act = activation_storage['value'].cpu().numpy()[0]   # (C, H, W)
    grad = gradient_storage['value'].cpu().numpy()[0]    # (C, H, W)

    # Ponderi = media globală a gradienților pe fiecare canal
    weights = np.mean(grad, axis=(1, 2))  # (C,)

    # Combinăm liniar activările
    cam = np.zeros(act.shape[1:], dtype=np.float32)  # (H, W)
    for i, w in enumerate(weights):
        cam += w * act[i]

    # ReLU + normalizare
    cam = np.maximum(cam, 0)
    if cam.max() > 1e-8:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam)

    # Redimensionăm la dimensiunile imaginii originale
    orig_w, orig_h = image_pil.size
    cam_resized = cv2.resize(cam, (orig_w, orig_h))

    # Convertim heatmap-ul la culori JET
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Suprapunem heatmap-ul peste imaginea originală
    orig_array = np.array(image_pil.convert('RGB'))
    superimposed = cv2.addWeighted(orig_array, 0.5, heatmap, 0.5, 0)

    result_image = Image.fromarray(superimposed)
    return result_image, CLASSES[target_class_idx]


class GradCamView(APIView):
    """
    Endpoint Grad-CAM: primește o imagine (preferabil decupată cu bbox YOLO),
    rulează Grad-CAM pe ResNet și returnează imaginea cu heatmap suprapus.
    """
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        summary="Grad-CAM pe imagine decupată (ResNet)",
        description=(
            "Primește o imagine MRI (de preferință decupată cu bounding box din YOLO) "
            "și returnează imaginea cu heatmap Grad-CAM suprapus, generat de modelul ResNet. "
            "Returnează PNG cu heatmap + header X-GradCam-Class cu clasa prezisă."
        ),
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'image': {
                        'type': 'string',
                        'format': 'binary',
                        'description': 'Imaginea MRI (decupată sau originală)'
                    }
                },
                'required': ['image']
            }
        },
        responses={
            200: OpenApiTypes.BINARY,
            400: OpenApiTypes.OBJECT,
            500: OpenApiTypes.OBJECT
        }
    )
    def post(self, request):
        from rest_framework.response import Response

        if 'image' not in request.FILES:
            return Response({"error": "No image provided"}, status=400)

        image_file = request.FILES['image']
        try:
            raw_image = Image.open(image_file)

            # Dacă imaginea are canal alpha (de ex. imagine segmentată SAM),
            # crop-uim la bounding box-ul zonei non-transparente (tumoarea).
            if raw_image.mode in ('RGBA', 'LA') or (raw_image.mode == 'P' and 'transparency' in raw_image.info):
                raw_rgba = raw_image.convert('RGBA')
                alpha_channel = np.array(raw_rgba.split()[-1])  # canalul A
                # Găsim pixelii cu alpha > 10 (zona tumorii)
                rows = np.any(alpha_channel > 10, axis=1)
                cols = np.any(alpha_channel > 10, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    # Adăugăm un mic padding de 5px
                    pad = 5
                    rmin = max(0, rmin - pad)
                    rmax = min(alpha_channel.shape[0], rmax + pad)
                    cmin = max(0, cmin - pad)
                    cmax = min(alpha_channel.shape[1], cmax + pad)
                    raw_image = raw_rgba.crop((cmin, rmin, cmax, rmax))
                    print(f"[GradCam] Crop RGBA la bbox tumoare: ({cmin},{rmin},{cmax},{rmax})")
                else:
                    raw_image = raw_rgba
                    print("[GradCam] Alpha channel found but all transparent - using full image")

            # Salvăm masca alpha (dacă există) pentru a o aplica după Grad-CAM
            # Astfel culorile heatmap vor apărea DOAR pe zona tumorii
            alpha_mask = None
            if raw_image.mode == 'RGBA':
                alpha_mask = raw_image.split()[3]  # canal alpha (Image mode 'L')
                background = Image.new('RGB', raw_image.size, (255, 255, 255))
                background.paste(raw_image, mask=alpha_mask)
                image_pil = background
            else:
                image_pil = raw_image.convert('RGB')
            print(f"[GradCam] Dimensiune imagine după crop: {image_pil.size}")
        except Exception as e:
            return Response({"error": f"Invalid image: {e}"}, status=400)

        # Modelul ResNet este deja în cache din apelul CNN voting
        # Încearcă 'resnet101' (poate fi de fapt ResNet34/50 după fallback)
        model_name = 'resnet101'

        try:
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
                print(f"[GradCam] Model {model_name} încărcat din {full_path}")
            else:
                print(f"[GradCam] Folosesc modelul {model_name} din cache.")

            model = LOADED_MODELS[model_name]
            print(f"[GradCam] Tip model: {type(model).__name__}")

            gradcam_image, predicted_class = generate_gradcam(model, image_pil)
            print(f"[GradCam] Clasă prezisă: {predicted_class}")

            # Aplicăm masca alpha pe rezultatul Grad-CAM
            # Exteriorul tumorii devine transparent → culorile heatmap apar DOAR pe tumoare
            if alpha_mask is not None:
                # Convertim rezultatul RGB la RGBA și aplicăm masca
                gradcam_rgba = gradcam_image.convert('RGBA')
                # Redimensionăm masca dacă dimensiunile diferă (din cauza padding-ului)
                if alpha_mask.size != gradcam_rgba.size:
                    alpha_mask = alpha_mask.resize(gradcam_rgba.size, Image.LANCZOS)
                gradcam_rgba.putalpha(alpha_mask)
                gradcam_image = gradcam_rgba
                print("[GradCam] Mască alpha aplicată pe heatmap")

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": f"Grad-CAM failed: {str(e)}"}, status=500)

        # Returnăm imaginea PNG
        buffer = io.BytesIO()
        gradcam_image.save(buffer, format='PNG')
        buffer.seek(0)

        response = HttpResponse(buffer.read(), content_type='image/png')
        response['X-GradCam-Class'] = predicted_class
        response['Access-Control-Expose-Headers'] = 'X-GradCam-Class'
        return response
