# coding: utf-8
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2  # add 20170825

cap = cv2.VideoCapture(0)  # add 20170825
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.  
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
# MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.  
PATH_TO_CKPT = 'training/frozen_inference_graph.pb/frozen_inference_graph.pb'
# PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.  
PATH_TO_LABELS = os.path.join('data', 'tv_vehicle_detection.pbtxt')
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 1

tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        frame_width = 640
        frame_height = 640
        out_video = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 10,
                                    (frame_width, frame_height))
        while True:  # for image_path in TEST_IMAGE_PATHS:    #changed 20170825
            ret, image = cap.read()
            if ret == True:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                height,width,_=np.shape(image)
                image_np=np.pad(image, ((80,80),(0, 0), (0, 0)), 'constant')#填充
                #image_np=image[92:390,172:470]
                #image_np =cv2.resize(image_np,(500,500))
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                cv2.imshow('object detection', image_np)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                # Write the frame into the file 'output.avi'
                out_video.write(image_np)
            else:
                print('===========error')
                break
out_video.release()
cap.release()
cv2.destroyAllWindows()

