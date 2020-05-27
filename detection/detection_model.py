def detection_main(input_name,dir_path):
    from imageai.Detection.Custom import CustomObjectDetection
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("detection/hololens/models/detection_model-ex-015--loss-0018.828.h5") 
    detector.setJsonPath("detection/hololens/json/detection_config.json")
    detector.loadModel()
    output_path = dir_path + "_output.png"
    detections = detector.detectObjectsFromImage(input_image= input_name, output_image_path= output_path)
    return (detections,output_path)
    #for detection in detections:

    
    #    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])