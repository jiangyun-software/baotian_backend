from django.urls import path
from . import views

#link urls to the views
urlpatterns = [
    #path('api/', views.detection_api),
    #path('', views.test),
    path('posts/', views.PostView.as_view(), name= 'posts_list'),
    path('', views.test),
]
