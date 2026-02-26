"""
URL configuration for djangoProject1 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from .views import TumorDetectionView, TumorDetectionJSONView, YoloVotingView, YoloVotingComplexView, YoloVotingLabelView, YoloVotingComplexLabelView
from .views_cnn_vit import TumorClassificationCNNView
from .views_cnn_vit_voting import CnnVitVotingLikelihoodView
from .views_cnn_vit_voting_label import CnnVitVotingLabelView
from .views_classification_report import ClassificationReportView, ClassificationReportCroppedView, ClassificationReportSamView
from .views_auto_annotate import AutoAnnotateView
from .views_gradcam import GradCamView
from .views_lime import LimeView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),

    # 3. ADAUGĂ RUTA PENTRU SWAGGER UI (Interfața vizuală)
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
   
    #Yolo
    path('api/detect-tumor/yolo/<str:version>/image', TumorDetectionView.as_view(), name='detect-tumor'),
    path('api/detect-tumor/yolo/<str:version>/json', TumorDetectionJSONView.as_view(), name='detect-tumor-json'),
    path('api/detect-tumor/yolo/voting-likelihood', YoloVotingView.as_view(), name='detect-tumor-voting'),
    path('api/detect-tumor/yolo/voting-complex/likelihood', YoloVotingComplexView.as_view(), name='detect-tumor-voting-complex'),
    path('api/detect-tumor/yolo/voting-label', YoloVotingLabelView.as_view(), name='detect-tumor-voting-label'),
    path('api/detect-tumor/yolo/voting-complex/label', YoloVotingComplexLabelView.as_view(), name='detect-tumor-voting-complex-label'),
    
    #Neuronal Network
    path('api/detect-tumor/neuronal-network/voting-likelihood', CnnVitVotingLikelihoodView.as_view(), name='detect-tumor-cnn-voting'),
    path('api/detect-tumor/neuronal-network/voting-label', CnnVitVotingLabelView.as_view(), name='detect-tumor-cnn-voting-label'),
    path('api/detect-tumor/neuronal-network/<str:model_type>', TumorClassificationCNNView.as_view(), name='detect-tumor-classification'),
    
    # Reports
    path('api/reports/classification-report', ClassificationReportView.as_view(), name='classification-report'),
    path('api/reports/classification-report-cropped', ClassificationReportCroppedView.as_view(), name='classification-report-cropped'),
    path('api/reports/classification-report-sam', ClassificationReportSamView.as_view(), name='classification-report-sam'),
    
    # Auto Annotation
    path('api/auto-annotate', AutoAnnotateView.as_view(), name='auto-annotate'),

    # Grad-CAM
    path('api/detect-tumor/grad-cam', GradCamView.as_view(), name='grad-cam'),

    # LIME
    path('api/detect-tumor/lime', LimeView.as_view(), name='lime'),
]
