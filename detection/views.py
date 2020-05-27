from .serializers import PostSerializer
from .models import Post
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import os
# Create your views here.
from .detection_model import detection_main
from django.http import JsonResponse,HttpResponse

class PostView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        posts_serializer = PostSerializer(data=request.data)
        if posts_serializer.is_valid():
            #保存到数据库
            posts_serializer.save()
            
            #获得本地图片地址
            image = posts_serializer.data["image"][1:]
            #获得图片名称
            dir_path,full_file_name = os.path.split(image)
            file_name, extension = os.path.splitext(full_file_name)


            
            #调用培训好的算法
            detection_result, output_image = detection_main(image,"media/output_images/"+file_name)
            
            #with open(output_image, 'rb') as f:
            #    image_data = f.read()
            response_url = os.path.join("http://127.0.0.1:8000/",output_image)
            
            return HttpResponse(response_url)
            # return HttpResponse(image_data, content_type="image/jpeg")
            # return Response(posts_serializer.data)
        else:
            print('error', posts_serializer.errors)
            return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)












"""
from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
import os
import shutil
from .detection_model import detection_main

from django.views.decorators.csrf import csrf_exempt

# Create your views here.
def detection_api(request):
    #get the GET parameters
    input_image = request.GET['image']

    #use the model to get the result
    os.chdir('C:\\Users\\DELL\\Desktop\\匠韵实习\\demo_server\\demo_server\\detection')
    detection_result = detection_main(input_image)


    #return image
    imagepath = os.path.join('C:\\Users\\DELL\\Desktop\\匠韵实习\\demo_server\\demo_server\\detection', file_name)  
    with open(imagepath, 'rb') as f:
        image_data = f.read()

    return HttpResponse(image_data, content_type="image/jpeg")


def test(request):
    return HttpResponse("test")


@csrf_exempt 
def upload(request):
    import json
    #data = json.loads(request.body.decode('utf-8'))
    data = request.data
    return HttpResponse(data)
"""