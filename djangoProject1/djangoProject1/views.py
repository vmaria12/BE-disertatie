import io
import numpy as np
from PIL import Image
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
# Importuri pentru Swagger
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes
from ultralytics import YOLO

# --- 1. Definim căile către modele ---
MODEL_PATHS = {
    'v8': r"D:\Disertatie\BE-disertatie\djangoProject1\Models\Yolo\yolo_v8.pt",
    'v9': r"D:\Disertatie\BE-disertatie\djangoProject1\Models\Yolo\yolo_v9.pt",
    'v12': r"D:\Disertatie\BE-disertatie\djangoProject1\Models\Yolo\yolo_v12.pt"
}

class TumorDetectionView(APIView):
    """Endpoint care returnează imaginea cu detecțiile desenate"""
    # <--- LINIA CRITICĂ 1: Aceste parsere permit upload-ul fizic
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        summary="Detectie Tumora (Upload Imagine)",
        description="Încarcă o imagine MRI. Selectează modelul de Yolo dorit prin URL: v8, v9 sau v12. Returnează imaginea cu detecțiile desenate.",
        parameters=[
            OpenApiParameter(
                name='version',
                type=str,
                location=OpenApiParameter.PATH,
                description='Versiunea modelului YOLO (v8, v9 sau v12)',
                required=True,
                enum=['v8', 'v9', 'v12']
            )
        ],
        # <--- LINIA CRITICĂ 2: Definim manual schema pentru a forța butonul de upload
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'image': {
                        'type': 'string',
                        'format': 'binary',  # Asta transformă câmpul în buton "Choose File"
                        'description': 'Selectează fișierul MRI'
                    }
                },
                'required': ['image']
            }
        },
        # Definim răspunsul ca fiind o imagine (nu JSON)
        responses={
            200: OpenApiTypes.BINARY,
            400: OpenApiTypes.OBJECT,
            500: OpenApiTypes.OBJECT
        }
    )
    def post(self, request, version, *args, **kwargs):
        # Verificăm dacă versiunea este validă
        if version not in MODEL_PATHS:
            return HttpResponse(
                f"Versiune invalidă: {version}. Folosește v8, v9 sau v12",
                status=400
            )

        # Încărcăm modelul corespunzător
        try:
            model_path = MODEL_PATHS[version]
            model = YOLO(model_path)
        except Exception as e:
            return HttpResponse(
                f"Eroare la încărcarea modelului {version}: {str(e)}",
                status=500
            )

        # Verificăm dacă există fișierul în request.FILES
        if 'image' not in request.FILES:
            return HttpResponse("Trebuie să încarci o imagine (key='image')", status=400)

        try:
            # 1. Preluăm imaginea
            uploaded_file = request.FILES['image']
            original_image = Image.open(uploaded_file)

            # 2. Predictie
            results = model.predict(original_image, conf=0.25)

            # 3. Procesare rezultat (BGR -> RGB)
            result_array_bgr = results[0].plot()
            result_array_rgb = result_array_bgr[..., ::-1]
            result_image = Image.fromarray(result_array_rgb)

            # 4. Salvare în buffer (RAM) ca PNG
            buffer = io.BytesIO()
            result_image.save(buffer, format="PNG")
            buffer.seek(0)

            # 5. Returnare
            return HttpResponse(buffer, content_type="image/png")

        except Exception as e:
            return HttpResponse(f"Eroare la procesare: {str(e)}", status=500)


class TumorDetectionJSONView(APIView):
    """Endpoint care returnează rezultatele ca JSON"""
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        summary="Detectie Tumora (Răspuns JSON)",
        description="Încarcă o imagine MRI și primești un răspuns JSON cu rezultatele detecției: dacă există tumoare și procentajul de încredere.",
        parameters=[
            OpenApiParameter(
                name='version',
                type=str,
                location=OpenApiParameter.PATH,
                description='Versiunea modelului YOLO (v8, v9 sau v12)',
                required=True,
                enum=['v8', 'v9', 'v12']
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
                    'tumoare_detectata': {
                        'type': 'boolean',
                        'description': 'True dacă a fost detectată tumoare, False altfel'
                    },
                    'numar_detecții': {
                        'type': 'integer',
                        'description': 'Numărul total de tumori detectate'
                    },
                    'detecții': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'clasa': {'type': 'string'},
                                'confidence': {'type': 'number'},
                                'confidence_procent': {'type': 'string'},
                                'bounding_box': {
                                    'type': 'object',
                                    'properties': {
                                        'x1': {'type': 'number'},
                                        'y1': {'type': 'number'},
                                        'x2': {'type': 'number'},
                                        'y2': {'type': 'number'}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            400: OpenApiTypes.OBJECT,
            500: OpenApiTypes.OBJECT
        }
    )
    def post(self, request, version, *args, **kwargs):
        # Verificăm dacă versiunea este validă
        if version not in MODEL_PATHS:
            return Response(
                {"error": f"Versiune invalidă: {version}. Folosește v8, v9 sau v12"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Încărcăm modelul corespunzător
        try:
            model_path = MODEL_PATHS[version]
            model = YOLO(model_path)
        except Exception as e:
            return Response(
                {"error": f"Eroare la încărcarea modelului {version}: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Verificăm dacă există fișierul în request.FILES
        if 'image' not in request.FILES:
            return Response(
                {"error": "Trebuie să încarci o imagine (key='image')"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 1. Preluăm imaginea
            uploaded_file = request.FILES['image']
            original_image = Image.open(uploaded_file)

            # 2. Predictie
            results = model.predict(original_image, conf=0.25)
            
            # 3. Extragem informațiile din rezultate
            detections = results[0].boxes
            
            # Construim lista de detecții
            detectii_lista = []
            for box in detections:
                # Extragem coordonatele bounding box-ului
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Extragem confidence și clasa
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                detectii_lista.append({
                    "clasa": class_name,
                    "confidence": round(confidence, 4),
                    "confidence_procent": f"{round(confidence * 100, 2)}%",
                    "bounding_box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })
            
            # 4. Construim răspunsul
            response_data = {
                "tumoare_detectata": len(detectii_lista) > 0,
                "numar_detecții": len(detectii_lista),
                "detecții": detectii_lista
            }
            
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Eroare la procesare: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class YoloVotingView(APIView):
    """Endpoint care rulează v8, v9, v12 și votează."""
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        summary="Detectie Tumora (Voting Ensemble)",
        description="Rulează YOLO v8, v9 și v12 pe aceeași imagine. Însumează probabilitățile detecțiilor pentru fiecare clasă și decide clasa finală.",
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
            200: OpenApiTypes.OBJECT,
            400: OpenApiTypes.OBJECT,
            500: OpenApiTypes.OBJECT
        }
    )
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response(
                {"error": "Trebuie să încarci o imagine (key='image')"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            uploaded_file = request.FILES['image']
            original_image = Image.open(uploaded_file)
            
            # Dicționare pentru rezultate
            model_results = {}
            class_scores = {} # { "Meningioma": 2.45, "Glioma": 0.1 }

            # Iterăm prin toate modelele
            for version, model_path in MODEL_PATHS.items():
                try:
                    model = YOLO(model_path)
                    results = model.predict(original_image, conf=0.25)
                    detections = results[0].boxes
                    
                    current_model_detections = []
                    
                    for box in detections:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        # Adăugăm la lista modelului curent
                        current_model_detections.append({
                            "clasa": class_name,
                            "confidence": round(confidence, 4),
                            "confidence_procent": f"{round(confidence * 100, 2)}%"
                        })
                        
                        # Adăugăm la scorul global (Voting)
                        if class_name in class_scores:
                            class_scores[class_name] += confidence
                        else:
                            class_scores[class_name] = confidence
                            
                    model_results[version] = current_model_detections
                    
                except Exception as e:
                    model_results[version] = {"error": str(e)}

            # Determinăm câștigătorul
            if not class_scores:
                winning_class = "Nu s-a detectat tumoare"
                max_score = 0
            else:
                winning_class = max(class_scores, key=class_scores.get)
                max_score = class_scores[winning_class]

            response_data = {
                "individual_results": model_results,
                "voting_result": {
                    "winning_class": winning_class,
                    "total_score": round(max_score, 4),
                    "all_scores": {k: round(v, 4) for k, v in class_scores.items()}
                }
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Eroare la procesare voting: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


import base64
import io
from PIL import ImageDraw, ImageFont
import json

class YoloVotingComplexView(APIView):
    """
    Endpoint nou care:
    1. Rulează v8, v9, v12.
    2. Calculează Voting (suma scorurilor).
    3. Identifică cea mai bună detecție.
    4. Desenează acea detecție pe imagine.
    5. Returnează IMAGINEA (PNG) pentru vizualizare directă în Swagger.
    6. Returnează datele JSON (scoruri, detecții) în header-ul 'X-Voting-Data'.
    """
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        summary="Detectie Tumora (Voting + Imagine)",
        description="Returnează imaginea procesată (PNG). Datele detaliate (JSON) sunt incluse în header-ul 'X-Voting-Data'.",
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
            200: OpenApiTypes.BINARY,
            400: OpenApiTypes.OBJECT,
            500: OpenApiTypes.OBJECT
        }
    )
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response(
                {"error": "Trebuie să încarci o imagine (key='image')"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            uploaded_file = request.FILES['image']
            original_image = Image.open(uploaded_file).convert("RGB")
            
            # Structuri de date
            model_results = {}
            class_scores = {} 
            all_detections_flat = []

            # 1. Rulare Modele
            for version, model_path in MODEL_PATHS.items():
                try:
                    model = YOLO(model_path)
                    results = model.predict(original_image, conf=0.25)
                    detections = results[0].boxes
                    
                    current_model_detections = []
                    
                    for box in detections:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        # Coordonate
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_coords = {
                            "x1": round(x1, 2), "y1": round(y1, 2),
                            "x2": round(x2, 2), "y2": round(y2, 2)
                        }

                        det_obj = {
                            "clasa": class_name,
                            "confidence": round(confidence, 4),
                            "confidence_procent": f"{round(confidence * 100, 2)}%",
                            "bounding_box": box_coords,
                            "model": version
                        }
                        current_model_detections.append(det_obj)
                        all_detections_flat.append(det_obj)
                        
                        # Voting Score Aggregation
                        if class_name in class_scores:
                            class_scores[class_name] += confidence
                        else:
                            class_scores[class_name] = confidence
                            
                    model_results[version] = current_model_detections
                    
                except Exception as e:
                    model_results[version] = {"error": str(e)}

            # 2. Determinare Câștigător Voting
            if not class_scores:
                winning_class = "Nu s-a detectat tumoare"
                max_score = 0
                best_detection = None
            else:
                winning_class = max(class_scores, key=class_scores.get)
                max_score = class_scores[winning_class]
                
                # Threshold check
                if max_score < 0.5:
                    winning_class = "Nu s-a detectat tumoare"
                    best_detection = None
                else:
                    # 3. Găsire "Cea mai bună detecție"
                    winning_candidates = [d for d in all_detections_flat if d['clasa'] == winning_class]
                    
                    if winning_candidates:
                        best_detection = max(winning_candidates, key=lambda x: x['confidence'])
                    else:
                        best_detection = None

            # 4. Desenare pe Imagine
            processed_image = original_image.copy()
            if best_detection:
                draw = ImageDraw.Draw(processed_image)
                coords = best_detection['bounding_box']
                x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
                
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                text = f"{best_detection['clasa']} {best_detection['confidence_procent']}"
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x1, y1), text, fill="white", font=font)

            # 5. Salvare Imagine în Buffer
            buffer = io.BytesIO()
            processed_image.save(buffer, format="PNG")
            buffer.seek(0)

            # 6. Pregătire Date JSON pentru Header
            json_data = {
                "individual_results": model_results,
                "voting_result": {
                    "winning_class": winning_class,
                    "total_score": round(max_score, 4),
                    "all_scores": {k: round(v, 4) for k, v in class_scores.items()}
                },
                "best_detection": best_detection
            }
            
            # Serializare JSON și codare Base64 pentru siguranță în header
            json_str = json.dumps(json_data)
            json_b64 = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
            response = HttpResponse(buffer, content_type="image/png")
            response['X-Voting-Data'] = json_b64
            return response

        except Exception as e:
            return Response(
                {"error": f"Eroare la procesare complexă: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class YoloVotingLabelView(APIView):
    """Endpoint care rulează v8, v9, v12 și votează pe baza numărului de etichete."""
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        summary="Detectie Tumora (Voting Label)",
        description="Rulează YOLO v8, v9 și v12. Numără aparițiile fiecărei clase detectate și decide clasa finală pe baza majorității.",
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
            200: OpenApiTypes.OBJECT,
            400: OpenApiTypes.OBJECT,
            500: OpenApiTypes.OBJECT
        }
    )
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response(
                {"error": "Trebuie să încarci o imagine (key='image')"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            uploaded_file = request.FILES['image']
            original_image = Image.open(uploaded_file)
            
            model_results = {}
            class_counts = {}
            class_probs_sum = {}

            for version, model_path in MODEL_PATHS.items():
                try:
                    model = YOLO(model_path)
                    results = model.predict(original_image, conf=0.25)
                    detections = results[0].boxes
                    
                    current_model_detections = []
                    
                    for box in detections:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        current_model_detections.append({
                            "clasa": class_name,
                            "confidence": round(confidence, 4),
                            "confidence_procent": f"{round(confidence * 100, 2)}%"
                        })
                        
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                            class_probs_sum[class_name] += confidence
                        else:
                            class_counts[class_name] = 1
                            class_probs_sum[class_name] = confidence
                            
                    model_results[version] = current_model_detections
                    
                    if not current_model_detections:
                        class_name = "Nu s-a detectat tumoare"
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                            class_probs_sum[class_name] += 1.0
                        else:
                            class_counts[class_name] = 1
                            class_probs_sum[class_name] = 1.0
                    
                except Exception as e:
                    model_results[version] = {"error": str(e)}

            if not class_counts:
                winning_class = "Nu s-a detectat tumoare"
                max_count = 0
            else:
                max_votes = max(class_counts.values())
                candidates = [c for c, v in class_counts.items() if v == max_votes]
                
                if len(candidates) == 1:
                    winning_class = candidates[0]
                else:
                    # Tie detected - use sum of probabilities
                    winning_class = max(candidates, key=lambda c: class_probs_sum.get(c, 0))
                
                max_count = class_counts[winning_class]

            response_data = {
                "individual_results": model_results,
                "voting_result": {
                    "winning_class": winning_class,
                    "vote_count": max_count,
                    "all_counts": class_counts,
                    "probability_sums": {k: round(v, 4) for k, v in class_probs_sum.items()}
                }
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Eroare la procesare voting label: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class YoloVotingComplexLabelView(APIView):
    """
    Endpoint nou care:
    1. Rulează v8, v9, v12.
    2. Calculează Voting pe baza numărului de etichete (Label Voting).
    3. Identifică cea mai bună detecție pentru clasa câștigătoare.
    4. Desenează acea detecție pe imagine.
    5. Returnează IMAGINEA (PNG).
    6. Returnează datele JSON în header-ul 'X-Voting-Data'.
    """
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        summary="Detectie Tumora (Voting Label + Imagine)",
        description="Returnează imaginea procesată (PNG) folosind logica de votare pe etichete. Datele detaliate (JSON) sunt incluse în header-ul 'X-Voting-Data'.",
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
            200: OpenApiTypes.BINARY,
            400: OpenApiTypes.OBJECT,
            500: OpenApiTypes.OBJECT
        }
    )
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response(
                {"error": "Trebuie să încarci o imagine (key='image')"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            uploaded_file = request.FILES['image']
            original_image = Image.open(uploaded_file).convert("RGB")
            
            model_results = {}
            class_counts = {}
            class_probs_sum = {}
            all_detections_flat = []

            # 1. Rulare Modele
            for version, model_path in MODEL_PATHS.items():
                try:
                    model = YOLO(model_path)
                    results = model.predict(original_image, conf=0.25)
                    detections = results[0].boxes
                    
                    current_model_detections = []
                    
                    for box in detections:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        # Coordonate
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_coords = {
                            "x1": round(x1, 2), "y1": round(y1, 2),
                            "x2": round(x2, 2), "y2": round(y2, 2)
                        }

                        det_obj = {
                            "clasa": class_name,
                            "confidence": round(confidence, 4),
                            "confidence_procent": f"{round(confidence * 100, 2)}%",
                            "bounding_box": box_coords,
                            "model": version
                        }
                        current_model_detections.append(det_obj)
                        all_detections_flat.append(det_obj)
                        
                        # Label Voting Aggregation
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                            class_probs_sum[class_name] += confidence
                        else:
                            class_counts[class_name] = 1
                            class_probs_sum[class_name] = confidence
                            
                    model_results[version] = current_model_detections
                    
                    if not current_model_detections:
                        class_name = "Nu s-a detectat tumoare"
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                            class_probs_sum[class_name] += 1.0
                        else:
                            class_counts[class_name] = 1
                            class_probs_sum[class_name] = 1.0
                    
                except Exception as e:
                    model_results[version] = {"error": str(e)}

            # 2. Determinare Câștigător Voting (Label)
            if not class_counts:
                winning_class = "Nu s-a detectat tumoare"
                max_count = 0
                best_detection = None
            else:
                max_votes = max(class_counts.values())
                candidates = [c for c, v in class_counts.items() if v == max_votes]
                
                if len(candidates) == 1:
                    winning_class = candidates[0]
                else:
                    # Tie detected - use sum of probabilities
                    winning_class = max(candidates, key=lambda c: class_probs_sum.get(c, 0))
                
                max_count = class_counts[winning_class]
                
                # 3. Găsire "Cea mai bună detecție" pentru clasa câștigătoare
                winning_candidates = [d for d in all_detections_flat if d['clasa'] == winning_class]
                
                if winning_candidates:
                    best_detection = max(winning_candidates, key=lambda x: x['confidence'])
                else:
                    best_detection = None

            # 4. Desenare pe Imagine
            processed_image = original_image.copy()
            if best_detection:
                draw = ImageDraw.Draw(processed_image)
                coords = best_detection['bounding_box']
                x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
                
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=3) # Blue for label voting distinction
                
                text = f"{best_detection['clasa']} {best_detection['confidence_procent']}"
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                draw.rectangle(text_bbox, fill="blue")
                draw.text((x1, y1), text, fill="white", font=font)

            # 5. Salvare Imagine în Buffer
            buffer = io.BytesIO()
            processed_image.save(buffer, format="PNG")
            buffer.seek(0)

            # 6. Pregătire Date JSON pentru Header
            json_data = {
                "individual_results": model_results,
                "voting_result": {
                    "winning_class": winning_class,
                    "vote_count": max_count,
                    "all_counts": class_counts,
                    "probability_sums": {k: round(v, 4) for k, v in class_probs_sum.items()}
                },
                "best_detection": best_detection
            }
            
            # Serializare JSON și codare Base64
            json_str = json.dumps(json_data)
            json_b64 = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
            response = HttpResponse(buffer, content_type="image/png")
            response['X-Voting-Data'] = json_b64
            return response

        except Exception as e:
            return Response(
                {"error": f"Eroare la procesare complexă label: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )