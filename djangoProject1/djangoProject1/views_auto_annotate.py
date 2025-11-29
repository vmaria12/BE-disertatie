from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from ultralytics.data.annotator import auto_annotate
from ultralytics import SAM
from django.conf import settings
from drf_spectacular.utils import extend_schema
from .serializers import ImageUploadSerializer, AutoAnnotateResponseSerializer
import os
import shutil
import tempfile
import cv2
import numpy as np
import base64
import json

class AutoAnnotateView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        request=ImageUploadSerializer,
        responses={200: AutoAnnotateResponseSerializer},
        summary="Auto Annotate Image",
        description="Upload an image to generate YOLO segmentation annotations using SAM 2.1. Can optionally provide a 'bbox' (JSON [x1, y1, x2, y2]) to skip detection and segment specific area."
    )
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']
        bbox_str = request.data.get('bbox')
        
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
        sam_model_name = os.path.join(base_dir, 'Models', 'SAM', "sam2.1_b.pt")
        output_dir = os.path.join(temp_dir, 'labels')
        
        # Ensure SAM model path exists or let ultralytics handle it (it downloads if name is simple, but we have a path)
        # If path provided doesn't exist, it might error.
        if not os.path.exists(sam_model_name):
             # Fallback to just name if local path fails, or check relative
             pass 

        try:
            annotation_data = ""
            
            if bbox_str:
                # --- Path A: Use provided Bounding Box with SAM directly ---
                try:
                    bbox = json.loads(bbox_str) # Expecting [x1, y1, x2, y2]
                except json.JSONDecodeError:
                    return Response({'error': 'Invalid bbox format. Expected JSON array [x1, y1, x2, y2]'}, status=status.HTTP_400_BAD_REQUEST)
                
                model = SAM(sam_model_name)
                results = model(image_path, bboxes=[bbox])
                
                # Process results to generate YOLO format annotation
                # YOLO format: class_id x_center y_center width height (for detection)
                # But for segmentation: class_id x1 y1 x2 y2 ... xn yn (normalized)
                
                if results and len(results) > 0:
                    result = results[0]
                    if result.masks:
                        # Get normalized segments
                        # result.masks.xyn returns list of arrays of normalized coordinates
                        segments = result.masks.xyn
                        
                        lines = []
                        for segment in segments:
                            # Class ID 0 by default for auto-annotation
                            line = "0 " + " ".join([f"{coord:.6f}" for point in segment for coord in point])
                            lines.append(line)
                        
                        annotation_data = "\n".join(lines)
                        
                        # Save to file for consistency (optional, but good for debugging)
                        os.makedirs(output_dir, exist_ok=True)
                        label_filename = os.path.splitext(image_file.name)[0] + '.txt'
                        with open(os.path.join(output_dir, label_filename), 'w') as f:
                            f.write(annotation_data)

            else:
                # --- Path B: Run Full Auto-Annotate (Detection + Segmentation) ---
                det_model_path = os.path.join(base_dir, 'Models', 'Yolo', 'yolo_v12.pt')
                
                # Verify model exists
                if not os.path.exists(det_model_path):
                     current_file_dir = os.path.dirname(os.path.abspath(__file__))
                     potential_path = os.path.join(os.path.dirname(current_file_dir), 'Models', 'Yolo', 'yolo_v12.pt')
                     if os.path.exists(potential_path):
                         det_model_path = potential_path
                     else:
                         # Try standard path if custom fails
                         det_model_path = 'yolo_v12.pt' # Let ultralytics find/download it

                auto_annotate(
                    data=images_dir,
                    det_model=det_model_path,
                    sam_model=sam_model_name,
                    output_dir=output_dir
                )
                
                # Read the generated label file
                label_filename = os.path.splitext(image_file.name)[0] + '.txt'
                label_path = os.path.join(output_dir, label_filename)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        annotation_data = f.read()

            # --- Visualization (Common for both paths) ---
            img = cv2.imread(image_path)
            if img is None:
                 return Response({'error': 'Could not read image for processing'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            height, width, _ = img.shape
            label_filename = os.path.splitext(image_file.name)[0] + '.txt' # Ensure filename is set
            
            if annotation_data:
                lines = annotation_data.strip().split('\n')
                for line in lines:
                    if not line.strip(): continue
                    parts = line.strip().split()
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

            # Create segmented image (black background, only tumor)
            segmented_img_base64 = None
            if annotation_data:
                mask = np.zeros_like(img)
                # Re-parse points for mask creation (we need to do this again or store them)
                lines = annotation_data.strip().split('\n')
                all_points = []
                for line in lines:
                    if not line.strip(): continue
                    parts = line.strip().split()
                    coords = [float(x) for x in parts[1:]]
                    points = np.array(coords).reshape(-1, 2)
                    points[:, 0] *= width
                    points[:, 1] *= height
                    points = points.astype(np.int32)
                    all_points.append(points)
                
                if all_points:
                    cv2.fillPoly(mask, all_points, (255, 255, 255))
                    masked_img = cv2.bitwise_and(cv2.imread(image_path), mask) # Use original image, not the one with green overlay
                    _, buffer_seg = cv2.imencode('.png', masked_img)
                    segmented_img_base64 = base64.b64encode(buffer_seg).decode('utf-8')
            
            return Response({
                'filename': label_filename,
                'annotation': annotation_data,
                'image_base64': img_base64,
                'segmented_image_base64': segmented_img_base64,
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
