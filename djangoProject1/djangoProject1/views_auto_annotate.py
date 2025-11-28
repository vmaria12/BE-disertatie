import os
import shutil
import tempfile
import cv2
import numpy as np
import base64
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from ultralytics.data.annotator import auto_annotate
from django.conf import settings
from drf_spectacular.utils import extend_schema
from .serializers import ImageUploadSerializer, AutoAnnotateResponseSerializer

class AutoAnnotateView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        request=ImageUploadSerializer,
        responses={200: AutoAnnotateResponseSerializer},
        summary="Auto Annotate Image",
        description="Upload an image to generate YOLO segmentation annotations using SAM 2.1. Returns the annotation text and a visualized image."
    )
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']
        
        # Create a temporary directory for this request
        temp_dir = tempfile.mkdtemp()
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir)
        
        # Save the uploaded image
        image_path = os.path.join(images_dir, image_file.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        base_dir = settings.BASE_DIR
        det_model_path = os.path.join(base_dir, 'Models', 'Yolo', 'yolo_v12.pt')
        sam_model_name = "sam2.1_b.pt"
        output_dir = os.path.join(temp_dir, 'labels')

        # Verify model exists
        if not os.path.exists(det_model_path):
             current_file_dir = os.path.dirname(os.path.abspath(__file__))
             potential_path = os.path.join(os.path.dirname(current_file_dir), 'Models', 'Yolo', 'yolo_v12.pt')
             if os.path.exists(potential_path):
                 det_model_path = potential_path
             else:
                 shutil.rmtree(temp_dir)
                 return Response({'error': f'YOLO model not found at {det_model_path}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            # Run auto_annotate
            auto_annotate(
                data=images_dir,
                det_model=det_model_path,
                sam_model=sam_model_name,
                output_dir=output_dir
            )

            # Read the generated label file
            label_filename = os.path.splitext(image_file.name)[0] + '.txt'
            label_path = os.path.join(output_dir, label_filename)
            
            annotation_data = ""
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    annotation_data = f.read()

            # Process image for visualization
            img = cv2.imread(image_path)
            if img is None:
                 return Response({'error': 'Could not read image for processing'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            height, width, _ = img.shape
            
            if annotation_data:
                lines = annotation_data.strip().split('\n')
                for line in lines:
                    if not line.strip(): continue
                    parts = line.strip().split()
                    # class_id = int(parts[0]) # Not used for drawing currently
                    coords = [float(x) for x in parts[1:]]
                    
                    # Reshape to (N, 2)
                    points = np.array(coords).reshape(-1, 2)
                    
                    # Denormalize
                    points[:, 0] *= width
                    points[:, 1] *= height
                    
                    # Convert to int32 for cv2
                    points = points.astype(np.int32)
                    
                    # Draw polygon (Green)
                    cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    # Fill with semi-transparent color
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [points], (0, 255, 0))
                    alpha = 0.3
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # Encode to PNG base64
            _, buffer = cv2.imencode('.png', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return Response({
                'filename': label_filename,
                'annotation': annotation_data,
                'image_base64': img_base64,
                'message': 'Annotation generated successfully.' if annotation_data else 'No objects detected.'
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                pass
