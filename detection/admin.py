from django.contrib import admin
from .models import ImageUpload,SheetUpload,AnnotationImage

# Register your models here.
admin.site.register(SheetUpload)
admin.site.register(AnnotationImage)
admin.site.register(ImageUpload)