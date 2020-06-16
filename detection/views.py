from .serializers import ImageUploadSerializer,SheetUploadSerializer,AnnotationImageSerializer
from .models import AnnotationImage,ImageUpload,SheetUpload
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import os
# Create your views here.
from .sift_match import img_boundary_match
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
                print(annotation_images_serializer.validated_data["title"]+"已存在")
            else:
                annotation_images_serializer.save()
            
            return Response(annotation_images_serializer.data)
        else:
            print('error', annotation_images_serializer.errors)
            return Response(annotation_images_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#检测图片
sides_data = {
    "1H":{"start_point":(300,190),"end_point":(1150,425),"template":"detection/templates/right.png","min_count":4},
    "2H":{"start_point":(300,180),"end_point":(1400,500),"template":"detection/templates/right_new.png","min_count":4},
    "3H":{"start_point":(300,160),"end_point":(1050,315),"template":"detection/templates/right_3H.png","min_count":4},
    "4H":{"start_point":(170,160),"end_point":(1000,420),"template":"detection/templates/right.png","min_count":4},
    "5H":{"start_point":(250,210),"end_point":(930,420),"template":"detection/templates/right_5H.png","min_count":4},
    "6H":{"start_point":(250,120),"end_point":(1000,300),"template":"detection/templates/right_6H.png","min_count":10},
}


#上传图片调用算法
class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        images = ImageUpload.objects.all()
        serializer = ImageUploadSerializer(images, many=True)
        return Response(serializer.data)
    
    def post(self, request, *args, **kwargs):
        images_serializer = ImageUploadSerializer(data=request.data)
        if images_serializer.is_valid():
            if ImageUpload.objects.filter(title=images_serializer.validated_data["title"]).exists():
                print(images_serializer.validated_data["title"])
            else:
                images_serializer.save()
            
            input_path = os.path.join("media/upload_images/",images_serializer.validated_data["title"])
            save_path = "media/sift_images/"

            #调用培训好的算法
            side = images_serializer.validated_data["side"]
            datas,sift_imag =img_boundary_match(input_path,save_path,sides_data[side]["template"],sides_data[side]["start_point"],sides_data[side]["end_point"],sides_data[side]["min_count"])
            
            #打开图片并返回
            with open(sift_imag, 'rb') as f:
                image_data = f.read()
            return HttpResponse(image_data, content_type="image/png")
            # return Response(posts_serializer.data)
        else:
            print('error', images_serializer.errors)
            return Response(images_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

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








