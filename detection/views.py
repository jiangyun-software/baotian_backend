from .serializers import ImageUploadSerializer,SheetUploadSerializer,AnnotationImageSerializer,CroppedImageUploadSerializer
from .models import AnnotationImage,ImageUpload,SheetUpload,CroppedImageUpload
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import os
import shutil
from .sift_match import img_boundary_match
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .output_json import to_txt_files 
from .yolo_detect import predict

#服务器地址
server_url = "http://121.37.23.147:8004/"

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
    "0H":{"start_point":(303,193),"end_point":(1146,420),"template":"detection/templates/Template_0H.png","min_count":10},
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
            sift_img =img_boundary_match(input_path,save_path,cropped_path,sides_data[side]["template"],sides_data[side]["start_point"],sides_data[side]["end_point"],sides_data[side]["min_count"])
            
            #定义缺陷预测图片储存位置
            present_path,filename = os.path.split(sift_img)
            filename,extension = os.path.splitext(filename)
            detection_img_path = os.path.join('media/detection_images',filename+'_predicted.png')

            #调用培训好的缺陷检测算法
            #sift_img = os.path.join('/usr/src/app',sift_img)
            #defect_predict(sift_img,"checkp/model.ckpt",detection_img_path)

            #打开sift图片并返回
            with open(sift_img, 'rb') as f:
                image_data = f.read()
            return HttpResponse(image_data, content_type="image/png")
        else:
            print('error', images_serializer.errors)
            return Response(images_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class CroppedImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        cropped_images = CroppedImageUpload.objects.all()
        cropped_serializer = CroppedImageUploadSerializer(images, many=True)
        return Response(cropped_serializer.data)
    
    def post(self, request, *args, **kwargs):
        cropped_images_serializer = CroppedImageUploadSerializer(data=request.data)
        #判断上传图片是否已经存在，只保存数据库中不存在的图片
        #!需要再增加一个size判断
        
        if cropped_images_serializer.is_valid():
            cropped_img = os.path.join("media/sift_cropped_images/",cropped_images_serializer.validated_data["title"])
            cropped_images_serializer.save()
            #定义输入图片的路径与输出图片的文件夹
        
        
        #将图片移动到子文件夹,因为yolo算法默认对整个文件夹里的图片都进行操作，我们把它放到子文件夹以保证它单独运行
        filename,extension = os.path.splitext(cropped_images_serializer.validated_data["title"])
        dir_path = os.path.join("media/sift_cropped_images/",filename) #子文件夹路径
        if not os.path.exists(dir_path): #不存在子文件夹则创建新的子文件夹
            os.mkdir(dir_path)
        else:
            shutil.rmtree(dir_path) #存在则删掉再新建
            os.mkdir(dir_path)
        shutil.move(cropped_img,dir_path)
        

        #建立存放预测后照片的文件夹
        detection_img_path = os.path.join('media/detection_images',filename+'_predicted')
        if not os.path.exists(detection_img_path):
            os.mkdir(detection_img_path)
        else:
            shutil.rmtree(detection_img_path) #存在则删掉再新建
            os.mkdir(detection_img_path)

        #调用培训好的缺陷检测算法
        predict(dir_path,detection_img_path)

        #获得预测后图片的图片路径
        detection_filename = os.listdir(detection_img_path)[0]
        detection_predict_path = os.path.join(detection_img_path,detection_filename)
        #打开预测后图片并返回
        with open(detection_predict_path, 'rb') as f:
            image_data = f.read()
        return HttpResponse(image_data, content_type="image/png")
        return HttpResponse("succeed in yolo")
    #else:
    #    print('error', cropped_images_serializer.errors)
    #    return Response(cropped_images_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

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

 






