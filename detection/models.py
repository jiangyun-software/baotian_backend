
from django.db import models

# Create your models here.

class Post(models.Model):
    title = models.CharField(max_length=100)
    file_type = models.TextField()
    image = models.ImageField(upload_to='post_images')
    
    def __str__(self):
        return self.title