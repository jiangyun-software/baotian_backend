from .serializers import ImageUploadSerializer,SheetUploadSerializer,AnnotaionImageSerializer
from .models import AnnotationImage,ImageUpload,SheetUpload
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import os
# Create your views here.
from .detection_model import detection_main
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_exempt
from pprint import pprint
import json
from .output_json import to_txt_files 


server_url = "http://jyzn.nat300.top/"
annotation_path = './media/annotations/'

def test(request):
    return HttpResponse("succeed")

#接受标注数据
@csrf_exempt
def annotation(request):
    if request.method=="POST":
        res = request.FILES['file']
        res = res.read().decode('utf8')
        res = json.loads(res)
        to_txt_files(res,annotation_path)
    return HttpResponse("成功上传标记数据到"+server_url)

#接受标注图片
class AnnotationImageView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        annotation_images = ImageUpload.objects.all()
        serializer = AnnotationImageSerializer(annotation_images,many=True)
        return Response(serializer.data)
    
    def post(self, request, *args, **kwargs):
        annotation_images_serializer = AnnotationImageSerializer(data=request.data)
        if annotation_images_serializer.is_valid():
            #only save files that dont exits
            if AnnotationImage.objects.filter(title=annotation_images_serializer.validated_data["title"]).exists():
                print(annotation_images_serializer.validated_data["title"])
            else:
                annotation_images_serializer.save()
            
            return Response(annotation_images_serializer.data)
        else:
            print('error', annotation_images_serializer.errors)
            return Response(annotation_images_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#检测图片
class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        images = ImageUpload.objects.all()
        serializer = ImageUploadSerializer(images, many=True)
        return Response(serializer.data)
    
    def post(self, request, *args, **kwargs):
        images_serializer = ImageUploadSerializer(data=request.data)
        if images_serializer.is_valid():
            images_serializer.save()
            
            #获得本地图片地址
            image = images_serializer.data["image"][1:]
            
            #获得图片名称
            dir_path,full_file_name = os.path.split(image)
            file_name, extension = os.path.splitext(full_file_name)
            
            #调用培训好的算法
            detection_result, output_image = detection_main(image,"media/output_images/"+file_name)
            
            #with open(output_image, 'rb') as f:
            #    image_data = f.read()
            response_url = os.path.join(server_url,output_image)
            
            return HttpResponse(response_url)
            # return HttpResponse(image_data, content_type="image/jpeg")
            # return Response(posts_serializer.data)
        else:
            print('error', posts_serializer.errors)
            return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#表格数据预测
class SheetUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        sheets = SheetUpload.objects.all()
        serializer = SheetUploadSerializer(sheets, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        sheets_serializer = SheetUploadSerializer(data=request.data)
        if sheets_serializer.is_valid():
            #保存到数据库
            sheets_serializer.save()
            
            #获得表格
            sheet = sheets_serializer.data["sheet"][1:]
            
   
            return Response(sheets_serializer.data)
        else:
            print('error',sheets_serializer.errors)
            return Response(sheets_serializer.errors, status=status.HTTP_400_BAD_REQUEST)








