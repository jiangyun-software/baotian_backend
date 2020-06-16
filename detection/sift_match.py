import numpy as np
import cv2
import os
import glob
import shutil
import imutils
import argparse
import time


def img_boundary_match(file_path,save_path, template_path,start_point,end_point,MIN_MATCH_COUNT):
    orig_path = "./"
    img_template = cv2.imread(template_path,0)
    Count_num = 0
    # 1H img_template: right
    # 2H img_template: right
    # 3H img_template: right_3H
    # 4H img_template: right
    # 5H img_template: right
    # 6H img_template: right_6H
    for filename in glob.glob(file_path):
        print(filename)
        img2 = cv2.imread(filename,0)
        img0 = cv2.imread(filename,0)
        Count_num = Count_num+1
        # 使用SIFT检测角点
        sift = cv2.xfeatures2d.SIFT_create()
        #sift=cv2.SIFT()

        # 获取关键点和描述符
        kp1, des1 = sift.detectAndCompute(img_template,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # 定义FLANN匹配器
        index_params = dict(algorithm = 1, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # 使用KNN算法匹配
        matches = flann.knnMatch(des1,des2,k=2)

        # 去除错误匹配
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)


        # 单应性
        if len(good)>MIN_MATCH_COUNT:
            # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            # findHomography 函数是计算变换矩阵
            # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
            # 返回值：M 为变换矩阵，mask是掩模
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            # ravel方法将数据降维处理，最后并转换成列表格式
            matchesMask = mask.ravel().tolist()
            # 获取img2的图像尺寸
            h2,w2 = img2.shape
            # pts2是图像img2的四个顶点
            pts2 = np.float32([[0,0],[0,h2-1],[w2-1,h2-1],[w2-1,0]]).reshape(-1,1,2)
            # 计算变换后的四个顶点坐标位置
            dst2 = cv2.perspectiveTransform(pts2,M)
            # 计算角度
            
            # 获取img1的图像尺寸
            h,w = img_template.shape
            # pts是图像img1的四个顶点
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            # 计算变换后的四个顶点坐标位置
            dst = cv2.perspectiveTransform(pts,M)

            # 根据四个顶点坐标位置在img2图像画出变换后的边框
            img2 = cv2.polylines(img2,[np.int32(dst)],True,(255,0,0),3, cv2.LINE_AA)

        else:
            print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None

        # 显示匹配结果
        # 矩形框
        # 1H: 左上（300, 190），右下（1150, 425）
        # 2H: 左上（300, 180），右下（1400, 500）
        # 3H: 左上（300, 160），右下（1050, 315）
        # 4H: 左上（300, 180），右下（1400, 500）
        # 5H: 左上（300, 180），右下（1400, 500）
        # 6H：左上（250，120），右下（1000，300）

        # start_point = (300, 180)
        # end_point = (1400, 500) 
        color = (255,255,255)
        thickness = 3
        img2 = cv2.rectangle(img2, start_point, end_point, color, thickness)
        img0 = cv2.rectangle(img0, start_point, end_point, color, thickness)

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
        img3 = cv2.drawMatches(img_template,kp1,img2,kp2,good,None,**draw_params)
        datas = {'img_count_num':Count_num,
        'img_original':pts2.tolist(),
        'img_transformed':dst2.tolist(),
        'img_template_original':pts.tolist(),
        'img_template transformed':dst.tolist()}
        #plt.imshow(img0)
        #plt.show()
        #plt.imshow(img3)
        #plt.show()
        # "cv2.imshow" may lead to a Kernel crash so we change it to the 'plt' handle to check the image
        #cv2.imshow("Image show", img3)
        #cv2.waitKey(0)
        present_path,filename = os.path.split(file_path)
        filename,extension = os.path.splitext(filename)
        save_path = os.path.join(save_path,filename+'result.png')
        print(save_path)
        cv2.imwrite(save_path, img3)
        return (datas,save_path)
    print("=========================================================================================")
    print("All the images in the folder have been completed!!!!!!!!!!!")
    



# 矩形框
# 1H: start_point（300, 190），end_point（1150, 425）
# 2H: start_point（300, 180），end_point（1400, 500）
# 3H: start_point（300, 160），end_point（1050, 315）
# 4H: start_point（170, 160），end_point（1000, 420）
# 5H: start_point（250, 210），end_point（930, 420）
# 6H：start_point（250，120），end_point（1000，300）
#start_point = (250, 210)
#end_point = (930, 420)
# 6H：最小match是4个
# 1-5H：最小match是10个
#MIN_MATCH_COUNT = 10
#img_boundary_match("detection/CCD32020-5-26_18-43-15.png",'detection/templates/right_3H.png',(250, 210),(930, 420),10)