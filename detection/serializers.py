from rest_framework import serializers
from .models import ImageUpload,SheetUpload,AnnotationImage,CroppedImageUpload

class AnnotationImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnnotationImage
        fields = '__all__'

class SheetUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = SheetUpload
        fields = '__all__'

class ImageUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageUpload
        fields = '__all__'

class CroppedImageUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = CroppedImageUpload
        fields = '__all__'