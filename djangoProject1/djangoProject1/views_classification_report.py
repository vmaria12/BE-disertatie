import csv
import os
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings

class ClassificationReportView(APIView):
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
