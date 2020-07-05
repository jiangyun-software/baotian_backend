from django.contrib import admin
from .models import ImageUpload,SheetUpload,AnnotationImage,CroppedImageUpload

# Register your models here.
admin.site.register(SheetUpload)
admin.site.register(AnnotationImage)
admin.site.register(ImageUpload)
admin.site.register(CroppedImageUpload)