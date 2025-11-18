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