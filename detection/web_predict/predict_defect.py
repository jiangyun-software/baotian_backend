import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import os
from .utils import utils, helpers
from .builders import model_builder
from matplotlib import pyplot as plt
import sys


def defect_predict(input_path, model_path, output_path):
    os.chdir("detection/web_predict")
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=input_path, required=False, help='The image you want to predict on. ')
    parser.add_argument('--checkpoint_path', type=str, default=model_path, required=False, help='The path to the latest checkpoint weights for your model.')
    parser.add_argument('--crop_height', type=int, default=256, required=False, help='Height of cropped input image to network')
    parser.add_argument('--crop_width', type=int, default=256, required=False, help='Width of cropped input image to network')
    parser.add_argument('--model', type=str, default='MobileUNet', required=False, help='The model you are using')
    parser.add_argument('--dataset', type=str, default="AOI", required=False, help='The dataset you are using')
    args = parser.parse_known_args()[0]
    class_names_list, label_values = helpers.get_label_info("class_dict.csv")
    num_classes = len(label_values)
    print("\n***** Begin prediction *****")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)
    net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
    net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 
    network, _ = model_builder.build_model(args.model, net_input=net_input,
                                            num_classes=num_classes,
                                            crop_width=args.crop_width,
                                            crop_height=args.crop_height,
                                            is_training=False)

    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver(max_to_keep=1000)
    saver.restore(sess, model_path)
    loaded_image = utils.load_image(input_path)
    image_list, index_list = utils.center_crop(loaded_image, 128, 1)
    vis=np.zeros((768,2560,3))
    for j in range(len(image_list)):
           
        input_image = np.expand_dims(np.float32(image_list[j][:args.crop_height, :args.crop_width]),axis=0)/255.0
        st = time.time()
        output_image = sess.run(network,feed_dict={net_input:input_image})# what format
        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)
        run_time = time.time()-st
        y_val=index_list[j][1]
        x_val=index_list[j][0] 
        out_image = helpers.colour_code_segmentation(output_image, label_values)
        vis[(y_val-1)*256:y_val*256,(x_val-1)*256:x_val*256]=out_image

    os.chdir("/usr/src/app")
    cv2.imwrite(output_path,cv2.cvtColor(cv2.resize(np.uint8(vis),(1280,384)), cv2.COLOR_RGB2BGR))
    print("done")
    

#defect_predict("CCD12019-11-12_18-43-11.png","checkp/model.ckpt","test.png")
#print(os.getcwd())
#print(os.chdir('detection/web_predict/'))