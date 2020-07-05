from django.urls import path
from . import views

#link urls to the views
urlpatterns = [
    #path('api/', views.detection_api),
    #path('', views.test),
    path('annotation_image/', views.AnnotationImageView.as_view()),
    path('image/', views.ImageUploadView.as_view()),
    path('cropped_image/', views.CroppedImageUploadView.as_view()),
    path('spreadsheet/', views.SheetUploadView.as_view()),
    path('annotation/', views.annotation),
    path('', views.test),

]
