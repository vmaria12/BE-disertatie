from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField(help_text="Încarcă imaginea MRI pentru analiză")