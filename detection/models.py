
from django.db import models

# Create your models here.

class AnnotationImage(models.Model):
    title = models.CharField(max_length=100)
    file_size = models.CharField(max_length=100)
    image = models.ImageField(upload_to='annotation_images')
    
    def __str__(self):
        return self.title


class SheetUpload(models.Model):
    title = models.CharField(max_length=100)
    file_size = models.CharField(max_length=100)
    sheet = models.FileField(upload_to='upload_sheets')
    def __str__(self):
        return self.title

class ImageUpload(models.Model):
    title = models.CharField(max_length=100)
    file_size = models.CharField(max_length=100)
    image = models.FileField(upload_to='upload_images')
    def __str__(self):
        return self.title