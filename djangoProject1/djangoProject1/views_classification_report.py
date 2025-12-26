import csv
import os
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings

from drf_spectacular.utils import extend_schema
from .serializers import ClassificationReportItemSerializer

class ClassificationReportView(APIView):
    @extend_schema(
        responses={200: ClassificationReportItemSerializer(many=True)},
        summary="Get Classification Report (Full Image)",
        description="Returns the classification report data for full images."
    )
    def get(self, request):
        csv_path = os.path.join(settings.BASE_DIR, 'classification_report_imagine_completa.csv')
        
        data = []
        try:
            with open(csv_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append({
                        'file_name': row['file_name'],
                        'clasa_reala': row['clasa_reala'],
                        'clasa_detectata': row['clasa_detectata']
                    })
            return Response(data)
        except FileNotFoundError:
            return Response({'error': 'CSV file not found'}, status=404)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class ClassificationReportCroppedView(APIView):
    @extend_schema(
        responses={200: ClassificationReportItemSerializer(many=True)},
        summary="Get Classification Report (Cropped Image)",
        description="Returns the classification report data for cropped images."
    )
    def get(self, request):
        csv_path = os.path.join(settings.BASE_DIR, 'classification_report_imagine_decupata.csv')
        
        data = []
        try:
            with open(csv_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append({
                        'file_name': row['file_name'],
                        'clasa_reala': row['clasa_reala'],
                        'clasa_detectata': row['clasa_detectata']
                    })
            return Response(data)
        except FileNotFoundError:
            return Response({'error': 'CSV file not found'}, status=404)
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class ClassificationReportSamView(APIView):
    @extend_schema(
        responses={200: ClassificationReportItemSerializer(many=True)},
        summary="Get Classification Report (SAM Image)",
        description="Returns the classification report data for SAM images."
    )
    def get(self, request):
        csv_path = os.path.join(settings.BASE_DIR, 'classification_report_imagine_sam.csv')
        
        data = []
        try:
            with open(csv_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append({
                        'file_name': row['file_name'],
                        'clasa_reala': row['clasa_reala'],
                        'clasa_detectata': row['clasa_detectata']
                    })
            return Response(data)
        except FileNotFoundError:
            return Response({'error': 'CSV file not found'}, status=404)
        except Exception as e:
            return Response({'error': str(e)}, status=500)
