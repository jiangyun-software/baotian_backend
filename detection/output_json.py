import os
import json
def output_json(path,filename,data):
    """
    输入存储路径，文件名，数据
    生成以图片为单位的json标注数据
    
    """
    full_filename = filename+".txt"
    full_filename = os.path.join(path,full_filename)
    with open(full_filename, 'w') as outfile:
        json.dump(data, outfile)
    print("success"+filename)



def to_txt_files(data,path):
    """
    输入所有图片的json数据
    输出按图片分的txt格式的json数据
    """
    for key, data in data.items():
        output_json(path,key,data)