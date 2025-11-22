import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from drf_spectacular.utils import extend_schema, OpenApiTypes
from PIL import Image
from django.conf import settings
import os

# Importăm resursele din views_cnn_vit pentru a refolosi logica de încărcare
from .views_cnn_vit import MODEL_PATHS, CLASSES, load_efficientnet, load_pytorch_model, LOADED_MODELS

class CnnVitVotingLikelihoodView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        summary="Voting CNN/ViT (Suma Probabilităților) ==> JSON",
        description="Rulează EfficientNet, ResNet101 și ViT. Însumează probabilitățile pentru fiecare clasă și decide câștigătorul.",
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
                    'individual_results': {'type': 'object'},
                    'voting_result': {
                        'type': 'object',
                        'properties': {
                            'winning_class': {'type': 'string'},
                            'total_score': {'type': 'number'},
                            'all_scores': {'type': 'object'}
                        }
                    }
                }
            },
            400: OpenApiTypes.OBJECT,
            500: OpenApiTypes.OBJECT
        }
    )
    def post(self, request):
        if 'image' not in request.FILES:
            return Response({"error": "No image provided"}, status=400)

        image_file = request.FILES['image']
        try:
            # Încărcăm imaginea o singură dată
            original_image = Image.open(image_file).convert('RGB')
        except Exception:
            return Response({"error": "Invalid image"}, status=400)

        # Inițializăm scorurile globale
        # CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
        class_scores = {cls: 0.0 for cls in CLASSES}
        individual_results = {}

        # Iterăm prin cele 3 modele
        # Ordinea: efficientnet_b7, resnet101, vit_b16
        for model_name in ['efficientnet_b7', 'resnet101', 'vit_b16']:
            try:
                # 1. Încărcare Model (folosind cache-ul global din views_cnn_vit sau încărcând din nou)
                if model_name not in LOADED_MODELS:
                    path_candidate = MODEL_PATHS[model_name]
                    
                    # Logică de path (copiată/adaptată din views_cnn_vit pentru siguranță)
                    if not os.path.exists(path_candidate):
                        base_parent = os.path.dirname(os.path.dirname(settings.BASE_DIR))
                        full_path = os.path.join(base_parent, path_candidate)
                        if not os.path.exists(full_path):
                             full_path = os.path.join(settings.BASE_DIR, "Models", "CNN", os.path.basename(path_candidate))
                    else:
                        full_path = path_candidate

                    if model_name == 'efficientnet_b7':
                        LOADED_MODELS[model_name] = load_efficientnet(full_path)
                    else:
                        LOADED_MODELS[model_name] = load_pytorch_model(full_path, model_name)
                
                model = LOADED_MODELS[model_name]

                # 2. Predicție și obținere probabilități
                probs = [] # Va fi o listă de float-uri [p_glioma, p_meningioma, p_notumor, p_pituitary]

                if model_name == 'efficientnet_b7':
                    import tensorflow as tf
                    # Preprocesare specifică EfficientNet
                    img = original_image.resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)
                    
                    predictions = model.predict(img_array)
                    score_tensor = tf.nn.softmax(predictions[0])
                    probs = score_tensor.numpy().tolist() # Convertim la listă Python standard
                    
                else:
                    # PyTorch (ResNet, ViT)
                    import torch
                    from torchvision import transforms
                    
                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    
                    input_tensor = preprocess(original_image)
                    input_batch = input_tensor.unsqueeze(0)
                    
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    input_batch = input_batch.to(device)
                    
                    with torch.no_grad():
                        output = model(input_batch)
                    
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    probs = probabilities.cpu().numpy().tolist()

                # 3. Actualizare scoruri și salvare rezultat individual
                model_result_entry = {}
                
                # Mapăm probabilitățile la clase
                for idx, prob in enumerate(probs):
                    cls_name = CLASSES[idx]
                    # Adăugăm la suma globală
                    class_scores[cls_name] += prob
                    
                    # Salvăm pentru output individual
                    model_result_entry[cls_name] = f"{prob * 100:.2f}%"

                # Determinăm câștigătorul local pentru acest model (doar pentru afișare)
                local_max_idx = np.argmax(probs)
                local_winner = CLASSES[local_max_idx]
                local_conf = probs[local_max_idx] * 100

                individual_results[model_name] = {
                    "predicted_class": local_winner,
                    "confidence": f"{local_conf:.2f}%",
                    "all_probabilities": model_result_entry
                }

            except Exception as e:
                individual_results[model_name] = {"error": str(e)}
                print(f"Error running {model_name}: {e}")

        # 4. Determinare Câștigător Final (Voting)
        if not class_scores:
             winning_class = "Error"
             max_score = 0
        else:
            winning_class = max(class_scores, key=class_scores.get)
            max_score = class_scores[winning_class]

        # Formatăm scorurile finale pentru afișare
        formatted_scores = {k: f"{v:.4f}" for k, v in class_scores.items()}

        return Response({
            "individual_results": individual_results,
            "voting_result": {
                "winning_class": winning_class,
                "total_score_sum": f"{max_score:.4f}",
                "all_scores_sum": formatted_scores
            }
        })