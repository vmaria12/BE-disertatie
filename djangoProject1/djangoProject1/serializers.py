from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field
from drf_spectacular.types import OpenApiTypes

@extend_schema_field({"type": "string", "format": "binary"})
class UploadImageField(serializers.ImageField):
    """ImageField care apare ca file-upload (format: binary) în Swagger UI."""
    pass

class ImageUploadSerializer(serializers.Serializer):
    image = UploadImageField(help_text="Încarcă imaginea MRI pentru analiză")

class ClassificationReportItemSerializer(serializers.Serializer):
    file_name = serializers.CharField()
    clasa_reala = serializers.CharField()
    clasa_detectata = serializers.CharField()

class AutoAnnotateResponseSerializer(serializers.Serializer):
    filename = serializers.CharField()
    annotation = serializers.CharField()
    image_base64 = serializers.CharField(required=False, help_text="Base64 encoded image with visualized annotations")
    message = serializers.CharField(required=False)