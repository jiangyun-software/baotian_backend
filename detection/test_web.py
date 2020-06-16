def semantic_seg(input_path, output_path):
    import os,time,cv2, sys, math
    import tensorflow as tf
    import argparse
    import numpy as np
    from tensorflow.python.platform import gfile
    from utils import utils, helpers
    from builders import model_builder
    # Initializing network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)
    
    with gfile.FastGFile('model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        #print(graph_def)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    # ???????????
    sess.run(tf.global_variables_initializer())
    # ??
    net_input = sess.graph.get_tensor_by_name('net_input:0')
    net_output = sess.graph.get_tensor_by_name('net_output:0')
    op = sess.graph.get_tensor_by_name('optimization:0')

    input_image0 = cv2.imread(input_path)
    input_image1 = cv2.resize(input_image0,(512,512))
    input_image = np.expand_dims(np.float32(input_image1),axis=0)/255.0
    #st = time.time()
    output_image = sess.run(op, feed_dict={net_input:input_image})

    #run_times_list.append(time.time()-st)
    output_image1 = cv2.resize()
    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    out_v=np.uint8(out_vis_image)
    out_vis=cv2.resize(out_v,(1280,500))
    

    cv2.imwrite(output_path,cv2.cvtColor(out_vis, cv2.COLOR_RGB2BGR))
    #cv2.imwrite("%s/%s_gt.png"%("Test", file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

#try
#semantic_seg("CCD12019-11-12_18-40-09.png","a_try.png")
