import os
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from django.http import HttpResponse
from PIL import Image, ImageDraw, ImageFont
import io
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes

# --- 1. Definim căile către modele CNN + ViT ---
MODEL_PATHS = {
    'efficientnet_b7': r"BE-disertatie\djangoProject1\Models\CNN\efficientnet_b7.h5",
    'resnet101': r"BE-disertatie\djangoProject1\Models\CNN\resnet101_brain.pth",
    'vit_b16': r"BE-disertatie\djangoProject1\Models\CNN\vit_b16_brain.pth"
}

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Cache for loaded models
LOADED_MODELS = {}

def load_efficientnet(path):
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    return model

def predict_efficientnet(model, image):
    import tensorflow as tf
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_idx = np.argmax(score)
    confidence = 100 * np.max(score)
    return CLASSES[class_idx], confidence

def load_pytorch_model(path, model_type):
    import torch
    import torchvision.models as models
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Helper to create model architecture
    def create_model(arch):
        if arch == 'resnet101':
            m = models.resnet101(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, 4)
        elif arch == 'resnet50':
             m = models.resnet50(weights=None)
             m.fc = torch.nn.Linear(m.fc.in_features, 4)
        elif arch == 'resnet34':
             m = models.resnet34(weights=None)
             m.fc = torch.nn.Linear(m.fc.in_features, 4)
        elif arch == 'vit_b16':
            m = models.vit_b_16(weights=None)
            m.heads.head = torch.nn.Linear(m.heads.head.in_features, 4)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        return m.to(device)

    # 1. Load the file content first
    try:
        content = torch.load(path, map_location=device)
    except Exception as e:
        raise Exception(f"Failed to load file {path}: {e}")
    
    state_dict = None
    if isinstance(content, dict):
        state_dict = content
        # Handle module. prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        state_dict = new_state_dict
    else:
        # It's a full model
        model = content.to(device)
        model.eval()
        return model

    current_arch = model_type
    model = create_model(current_arch)
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Failed to load {current_arch} with strict=True: {e}")
        
        if model_type == 'resnet101':
            print("Attempting fallback to ResNet34 based on common file mismatch...")
            try:
                fallback_model = create_model('resnet34')
                fallback_model.load_state_dict(state_dict, strict=True)
                print("Fallback to ResNet34 successful!")
                model = fallback_model
            except RuntimeError as e2:
                 print(f"Fallback to ResNet34 failed: {e2}")
                 # Try ResNet50
                 try:
                    print("Attempting fallback to ResNet50...")
                    fallback_model = create_model('resnet50')
                    fallback_model.load_state_dict(state_dict, strict=True)
                    print("Fallback to ResNet50 successful!")
                    model = fallback_model
                 except RuntimeError as e3:
                     raise e
        else:
             raise e

    model.eval()
    return model

def predict_pytorch(model, image):
    import torch
    from torchvision import transforms
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_batch = input_batch.to(device)
    
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, cat_id = torch.max(probabilities, 0)
import os
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from django.http import HttpResponse
from PIL import Image, ImageDraw, ImageFont
import io
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes


MODEL_PATHS = {
    'efficientnet_b7': r"BE-disertatie\djangoProject1\Models\CNN\efficientnet_b7.h5",
    'resnet101': r"BE-disertatie\djangoProject1\Models\CNN\resnet101_brain.pth",
    'vit_b16': r"BE-disertatie\djangoProject1\Models\CNN\vit_b16_brain.pth"
}

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Cache for loaded models
LOADED_MODELS = {}

def load_efficientnet(path):
    import tensorflow as tf
    # Check if GPU is available for TF, though not strictly necessary for load
    model = tf.keras.models.load_model(path)
    return model

def predict_efficientnet(model, image):
    import tensorflow as tf
    # EfficientNetB7 input size is usually 600x600, but we changed it to 224x224 based on error
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis
    
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_idx = np.argmax(score)
    confidence = 100 * np.max(score)
    return CLASSES[class_idx], confidence

def load_pytorch_model(path, model_type):
    import torch
    import torchvision.models as models
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Helper to create model architecture
    def create_model(arch):
        if arch == 'resnet101':
            m = models.resnet101(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, 4)
        elif arch == 'resnet50':
             m = models.resnet50(weights=None)
             m.fc = torch.nn.Linear(m.fc.in_features, 4)
        elif arch == 'resnet34':
             m = models.resnet34(weights=None)
             m.fc = torch.nn.Linear(m.fc.in_features, 4)
        elif arch == 'vit_b16':
            m = models.vit_b_16(weights=None)
            m.heads.head = torch.nn.Linear(m.heads.head.in_features, 4)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        return m.to(device)

    # 1. Load the file content first
    try:
        content = torch.load(path, map_location=device)
    except Exception as e:
        raise Exception(f"Failed to load file {path}: {e}")
    
    state_dict = None
    if isinstance(content, dict):
        state_dict = content
        # Handle module. prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        state_dict = new_state_dict
    else:
        # It's a full model
        model = content.to(device)
        model.eval()
        return model

    current_arch = model_type
    model = create_model(current_arch)
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Failed to load {current_arch} with strict=True: {e}")
        
        # Fallback logic specifically for ResNet101 -> ResNet34/50 mismatch
        if model_type == 'resnet101':
            print("Attempting fallback to ResNet34 based on common file mismatch...")
            try:
                fallback_model = create_model('resnet34')
                fallback_model.load_state_dict(state_dict, strict=True)
                print("Fallback to ResNet34 successful!")
                model = fallback_model
            except RuntimeError as e2:
                 print(f"Fallback to ResNet34 failed: {e2}")
                 # Try ResNet50
                 try:
                    print("Attempting fallback to ResNet50...")
                    fallback_model = create_model('resnet50')
                    fallback_model.load_state_dict(state_dict, strict=True)
                    print("Fallback to ResNet50 successful!")
                    model = fallback_model
                 except RuntimeError as e3:
                     # Re-raise original error if fallbacks fail
                     raise e
        else:
             raise e

    model.eval()
    return model

def predict_pytorch(model, image):
    import torch
    from torchvision import transforms
    
    # Standard ImageNet normalization
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_batch = input_batch.to(device)
    
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, cat_id = torch.max(probabilities, 0)
    
    return CLASSES[cat_id.item()], confidence.item() * 100

class TumorClassificationCNNView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        summary="Clasificare Tumoare (CNN/ViT) ==> JSON",
        description="Încarcă o imagine MRI și primește un JSON cu clasa prezisă și acuratețea.",
        parameters=[
            OpenApiParameter(
                name='model_type',
                type=str,
                location=OpenApiParameter.PATH,
                description='Tipul modelului (efficientnet_b7, resnet101, vit_b16)',
                required=True,
                enum=list(MODEL_PATHS.keys())
            )
        ],
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'image': {
                        'type': 'string',
                        'format': 'binary',
                        'description': 'Selectează fișierul MRI'
                    }
                },
                'required': ['image']
            }
        },
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'tumor_type': {'type': 'string'},
                    'accuracy': {'type': 'string'},
                    'model': {'type': 'string'}
                }
            },
            400: OpenApiTypes.OBJECT,
            500: OpenApiTypes.OBJECT
        }
    )
    def post(self, request, model_type):
        if model_type not in MODEL_PATHS:
            return Response({"error": f"Model type '{model_type}' not supported. Choose from {list(MODEL_PATHS.keys())}"}, status=400)
            
        if 'image' not in request.FILES:
            return Response({"error": "No image provided"}, status=400)

        image_file = request.FILES['image']
        try:
            image = Image.open(image_file).convert('RGB')
        except Exception:
            return Response({"error": "Invalid image"}, status=400)

        # Load Model
        global LOADED_MODELS
        if model_type not in LOADED_MODELS:

            path_candidate = MODEL_PATHS[model_type]
            
            # Check if it exists as is (absolute or relative to cwd)
            if not os.path.exists(path_candidate):
                base_parent = os.path.dirname(os.path.dirname(settings.BASE_DIR)) # d:\Disertatie
                full_path = os.path.join(base_parent, path_candidate)
                
                if not os.path.exists(full_path):
                     # Try relative to BASE_DIR directly (maybe user moved it?)
                     full_path = os.path.join(settings.BASE_DIR, "Models", "CNN", os.path.basename(path_candidate))
                     
                     if not os.path.exists(full_path):
                         return Response({"error": f"Model file not found. Checked: {path_candidate}"}, status=500)
            else:
                full_path = path_candidate

            try:
                if model_type == 'efficientnet_b7':
                    LOADED_MODELS[model_type] = load_efficientnet(full_path)
                else:
                    LOADED_MODELS[model_type] = load_pytorch_model(full_path, model_type)
            except Exception as e:
                return Response({"error": f"Failed to load model: {str(e)}"}, status=500)
        
        model = LOADED_MODELS[model_type]
        
        # Predict
        try:
            if model_type == 'efficientnet_b7':
                label, accuracy = predict_efficientnet(model, image)
            else:
                label, accuracy = predict_pytorch(model, image)
        except Exception as e:
             return Response({"error": f"Prediction failed: {str(e)}"}, status=500)
             
        return Response({
            "tumor_type": label,
            "accuracy": f"{accuracy:.2f}%",
            "model": model_type
        })