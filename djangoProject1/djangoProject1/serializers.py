from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(help_text="Încarcă imaginea MRI pentru analiză")

class ClassificationReportItemSerializer(serializers.Serializer):
    file_name = serializers.CharField()
    clasa_reala = serializers.CharField()
    clasa_detectata = serializers.CharField()

class AutoAnnotateResponseSerializer(serializers.Serializer):
    filename = serializers.CharField()
    annotation = serializers.CharField()
    image_base64 = serializers.CharField(required=False, help_text="Base64 encoded image with visualized annotations")
    message = serializers.CharField(required=False)