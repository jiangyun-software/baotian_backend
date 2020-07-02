from .serializers import ImageUploadSerializer,SheetUploadSerializer,AnnotationImageSerializer
from .models import AnnotationImage,ImageUpload,SheetUpload
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import os
from .sift_match import img_boundary_match
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .output_json import to_txt_files 
from .web_predict.predict_defect import defect_predict

#服务器地址
server_url = "http://jyzn.nat300.top/"

#返回简单的test字符串，用于测试服务器是否成功运行
def test(request):
    return HttpResponse("succeed")

#接受标注数据,以文件名为单位存为一个个单独的文件在/media/annotations/
@csrf_exempt
def annotation(request):
    annotation_path = './media/annotations/'
    if request.method=="POST":
        res = request.FILES['file']
        res = res.read().decode('utf8')
        res = json.loads(res)
        to_txt_files(res,annotation_path)
    return HttpResponse("成功上传标记数据到"+server_url)

#接收标注图片，若在数据库中不存在，则将图片存入/media/annotation_images/,将图片信息存入数据库
class AnnotationImageView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        annotation_images = ImageUpload.objects.all()
        serializer = AnnotationImageSerializer(annotation_images,many=True)
        #返回所有标注图片的信息
        return Response(serializer.data)
    
    def post(self, request, *args, **kwargs):
        annotation_images_serializer = AnnotationImageSerializer(data=request.data)
        if annotation_images_serializer.is_valid():
            #判断标注图片是否已经存在，只保存数据库中不存在的图片
            if AnnotationImage.objects.filter(title=annotation_images_serializer.validated_data["title"]).exists():
                print(annotation_images_serializer.validated_data["title"]+"已存在")
            else:
                annotation_images_serializer.save()
            #返回成功上传的图片信息
            return Response(annotation_images_serializer.data)
        else:
            print('error', annotation_images_serializer.errors)
            return Response(annotation_images_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#零件每个面对所对应的不同数据参数
sides_data = {
    "1H":{"start_point":(303,193),"end_point":(1146,420),"template":"detection/templates/Template_1H.png","min_count":4},
    "2H":{"start_point":(276,172),"end_point":(1413,489),"template":"detection/templates/Template_2H.png","min_count":4},
    "3H":{"start_point":(292,147),"end_point":(1020,323),"template":"detection/templates/Template_3H.png","min_count":4},
    "4H":{"start_point":(172,164),"end_point":(987,379),"template":"detection/templates/Template_4H.png","min_count":4},
    "5H":{"start_point":(270,221),"end_point":(916,397),"template":"detection/templates/Template_5H.png","min_count":4},
    "6H":{"start_point":(245,131),"end_point":(982,289),"template":"detection/templates/Template_6H.png","min_count":10},
}

#接受上传图片，并调用定位算法
class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        images = ImageUpload.objects.all()
        serializer = ImageUploadSerializer(images, many=True)
        return Response(serializer.data)
    
    def post(self, request, *args, **kwargs):
        images_serializer = ImageUploadSerializer(data=request.data)
        #判断上传图片是否已经存在，只保存数据库中不存在的图片
        #!需要再增加一个size判断
        if images_serializer.is_valid():
            if ImageUpload.objects.filter(title=images_serializer.validated_data["title"]).exists():
                print(images_serializer.validated_data["title"])
            else:
                images_serializer.save()
            
            #定义输入图片的路径与输出图片的文件夹
            input_path = os.path.join("media/upload_images/",images_serializer.validated_data["title"])
            save_path = "media/sift_images/"
            cropped_path  = "media/sift_cropped_images"

            #调用培训好的定位算法
            side = images_serializer.validated_data["side"] #判断是哪一个面
            sift_img,croped_img =img_boundary_match(input_path,save_path,cropped_path,sides_data[side]["template"],sides_data[side]["start_point"],sides_data[side]["end_point"],sides_data[side]["min_count"])
            
            #调用培训好的检测算法
            sift_img = os.path.join('/usr/src/app',sift_img)
            defect_predict(sift_img,"checkp/model.ckpt","media/detection_images/test.png")

            #打开图片并返回
            with open("media/detection_images/test.png", 'rb') as f:
                image_data = f.read()
            return HttpResponse(image_data, content_type="image/png")
        else:
            print('error', images_serializer.errors)
            return Response(images_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#接收表格数据
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
            sheet = sheets_serializer.data["sheet"][1:]#去掉路径最开始的'/'
            #返回接收的表格信息
            return Response(sheets_serializer.data)
        else:
            print('error',sheets_serializer.errors)
            return Response(sheets_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

 






