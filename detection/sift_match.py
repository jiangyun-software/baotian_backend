import numpy as np
import cv2
import os
import glob
import shutil
import imutils
import argparse
import time


def img_boundary_match(file_path,save_path,cropped_path,template_path,start_point,end_point,MIN_MATCH_COUNT):
    orig_path = "./"
    template_image = cv2.imread(template_path,0)
    Count_num = 0
    # 1H template_image: right
    # 2H template_image: right
    # 3H template_image: right_3H
    # 4H template_image: right
    # 5H template_image: right
    # 6H template_image: right_6H
    for filename in glob.glob(file_path):
        print(filename)
        detect_image = cv2.imread(filename,0)
        img0 = cv2.imread(filename,0)
        Count_num = Count_num+1
        # 使用SIFT检测角点
        sift = cv2.xfeatures2d.SIFT_create()
        #sift=cv2.SIFT()

        # 获取关键点和描述符
        kp1, des1 = sift.detectAndCompute(template_image,None)
        kp2, des2 = sift.detectAndCompute(detect_image,None)

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
        if len(good) > MIN_MATCH_COUNT:
            # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
            src_pts_old = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            src_pts_list = []
            #转换坐标系为左上角坐标系
            for pnt in src_pts:
                src_pts_list.append(((pnt[0] + 900, pnt[1] + 227))) #900, 227

            src_pts = np.asarray(src_pts_list)
            src_pts = src_pts.reshape(-1, 1, 2)
            
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            
            # findHomography 函数是计算变换矩阵
            # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
            # 返回值：M 为变换矩阵，mask是掩模
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # ravel方法将数据降维处理，最后并转换成列表格式
            matchesMask = mask.ravel().tolist()

            src_pts_new_x = []
            src_pts_new_y = []
            dst_pts_new_x = []
            dst_pts_new_y = []
            count_pts = 0
            for k in range(len(src_pts)):
                if matchesMask[k]==1:
                    count_pts = count_pts + 1
                    src_pts_new_x.append(src_pts_old[k][0][0])
                    src_pts_new_y.append(src_pts_old[k][0][1])
                    dst_pts_new_x.append(dst_pts[k][0][0])
                    dst_pts_new_y.append(dst_pts[k][0][1])
            
            start = time.time()
            # 三角形面积最大
            s=0
            for i in range(count_pts-2):#x1不用遍历到最后一个数字
                for j in range(i+1,count_pts-1):
                    s1=src_pts_new_x[j]-src_pts_new_x[i]
                    s2=src_pts_new_y[j]-src_pts_new_y[i]#先求出x2-x1和y2-y1，避免后面重复计算
                    for k in range(j+1,count_pts):
                        s4=abs(s1*(src_pts_new_y[k]-src_pts_new_y[i])-(src_pts_new_x[k]-src_pts_new_x[i])*s2)/2
                        if s<s4:
                            s=s4
                            idx_0 = i
                            idx_1 = j
                            idx_2 = k
            
            end = time.time()
            print("time: " + str(end - start))
            src_pts_new = np.float32([[src_pts_new_x[idx_0],src_pts_new_y[idx_0]], [src_pts_new_x[idx_1],src_pts_new_y[idx_1]], [src_pts_new_x[idx_2],src_pts_new_y[idx_2]]])
            dst_pts_new = np.float32([[dst_pts_new_x[idx_0],dst_pts_new_y[idx_0]], [dst_pts_new_x[idx_1],dst_pts_new_y[idx_1]], [dst_pts_new_x[idx_2],dst_pts_new_y[idx_2]]])

            print("src_pts_new: ", src_pts_new)
            print("dst_pts_new: ", dst_pts_new)
            M_new = cv2.getAffineTransform(dst_pts_new, src_pts_new)

            affined_image = cv2.warpAffine(detect_image, M_new, (detect_image.shape[1], detect_image.shape[0]))
            
            #1H: (303, 193), (1146, 420)
            #2H: (276, 172), (1413, 489)
            #3H: (292, 147),  (1020, 323)
            #4H: (172, 164)， (987, 379)
            #5H: (270, 221)， (916, 397)
            #6H: (245, 131), (982, 289)

            #剪裁部分代码
            #affined_image_bb = cv2.rectangle(affined_image, (start_point[0], start_point[1]), (end_point[0], end_point[1]), (255,255,255), 2)
            #affined_image_crop = affined_image_bb[start_point[1]:end_point[1],start_point[0]:end_point[0]] #2H: (303, 191), (1383, 470)
            #save_path_crop = os.path.join(cropped_path,filename+'cropped.png')
            #cv2.imwrite(save_path_crop, affined_image_crop)  
            #定义输出路径
            present_path,filename = os.path.split(file_path)
            filename,extension = os.path.splitext(filename)
            save_path_locate = os.path.join(save_path,filename+'result.png')
            

            cv2.imwrite(save_path_locate, affined_image)   
            
            

        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None

        
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=matchesMask,  # draw only inliers
                        flags=2)
        img3 = cv2.drawMatches(template_image, kp1, detect_image, kp2, good, None, **draw_params)
        


        if len(good) > MIN_MATCH_COUNT:
            for k in range(len(dst_pts_new)):
                img4 = cv2.circle(img3,(dst_pts_new[k][0]+ np.float32(1280),dst_pts_new[k][1]),8,(255,255,255),-1)
                img5 = cv2.circle(img3,(src_pts_new[k][0],src_pts_new[k][1]),8,(0,0,255),-1)
        """        
            cv2.imwrite(save_path_init, img5)
            return (save_path_init)
        else:
            cv2.imwrite(save_path_crop, img3)
            return (save_path_crop)
        """
        return (save_path_locate) #save_path_locate

    print("for loop complete!!!!!!!!!!!")
    
    print("=========================================================================================")
    print("All the images in the folder have been completed!!!!!!!!!!!")
    