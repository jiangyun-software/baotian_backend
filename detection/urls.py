from django.urls import path
from . import views

#link urls to the views
urlpatterns = [
    #path('api/', views.detection_api),
    #path('', views.test),
    path('posts/', views.PostView.as_view(), name= 'posts_list'),
    path('uploads/', views.UploadImageView.as_view(), name= 'uploads_list'),
    path('annotation/', views.annotation),
    path('', views.test),

]
