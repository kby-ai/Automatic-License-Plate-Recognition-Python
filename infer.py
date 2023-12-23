# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import sys
import os
import tensorflow as tf
from deskew_image import *

PATH_DETECT_CKPT = 'models/detect.pb'
PATH_RECOG_CKPT = 'models/recog.pb'

char_maps = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
	# Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    return img
	
def load_detect_graph():

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = detection_graph.as_graph_def()
        with tf.io.gfile.GFile(PATH_DETECT_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)

    return detection_graph, sess

def load_recog_graph():

    recog_graph = tf.Graph()
    with recog_graph.as_default():
        od_graph_def = recog_graph.as_graph_def()
        with tf.io.gfile.GFile(PATH_RECOG_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=recog_graph)

    return recog_graph, sess

def recognition_vehicle(image):
	detect_image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	recog_image_tensor = recog_graph.get_tensor_by_name('image_tensor:0')
	predicted_chars = recog_graph.get_tensor_by_name('AttentionOcr_v1/predicted_chars:0')
	predicted_scores = recog_graph.get_tensor_by_name('AttentionOcr_v1/predicted_scores:0')

	aligned_img = cv2.resize(image, (300, 300), cv2.INTER_CUBIC)
	aligned_img_expanded = np.expand_dims(aligned_img, axis=0)

	(boxes, scores, classes, num) = detect_sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={detect_image_tensor: aligned_img_expanded})

	recog_results = [];
	recog_boxes = [];
	num_detection = int(num[0])
	for i in range(num_detection):
		class_id = classes[0][i]
		if class_id == 1:
			x1 = int(boxes[0][i][1] * 0.99 * image.shape[1])
			y1 = int(boxes[0][i][0] * 0.99 * image.shape[0])
			x2 = int(boxes[0][i][3] * 1.015 * image.shape[1])
			y2 = int(boxes[0][i][2] * 1.01 * image.shape[0])

			aligned_recog_img = np.zeros((150, 300,3), np.uint8)
			cropped_img = image[y1:y2, x1:x2]
			ratio_x = 150 / float(cropped_img.shape[1])
			ratio_y = 150 / float(cropped_img.shape[0])

			ratio_max = min(ratio_x, ratio_y)
			dst_w = int(cropped_img.shape[1] * ratio_max)
			dst_h = int(cropped_img.shape[0] * ratio_max)

			resized_aligned_recog_img = cv2.resize(cropped_img, (dst_w, dst_h), cv2.INTER_CUBIC)
			aligned_recog_img[:dst_h, :dst_w, :] = resized_aligned_recog_img

			deskewd_img = skew_correct(resized_aligned_recog_img)
			deskewd_img = skew_correct(deskewd_img)

			detectPlates(deskewd_img)


			aligned_recog_img[:dst_h, 150:150+dst_w, :] = deskewd_img

			cv2.imwrite('roi.jpg', deskewd_img)

			aligned_recog_img_expanded = np.expand_dims(aligned_recog_img, axis=0)

			(chars, scores) = recog_sess.run(
				[predicted_chars, predicted_scores],
				feed_dict={recog_image_tensor: aligned_recog_img_expanded})

			recog_text = ''
			for j in range(len(chars[0])):
				recog_text += char_maps[chars[0][j]]

			recog_results.append(recog_text)
			recog_boxes.append((x1, y1, x2, y2))

	return recog_results, recog_boxes

def recognition_license_plate(image):
	detect_image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	recog_image_tensor = recog_graph.get_tensor_by_name('image_tensor:0')
	predicted_chars = recog_graph.get_tensor_by_name('AttentionOcr_v1/predicted_chars:0')
	predicted_scores = recog_graph.get_tensor_by_name('AttentionOcr_v1/predicted_scores:0')

	aligned_img = cv2.resize(image, (300, 300), cv2.INTER_CUBIC)
	aligned_img_expanded = np.expand_dims(aligned_img, axis=0)

	(boxes, scores, classes, num) = detect_sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={detect_image_tensor: aligned_img_expanded})
	recog_results = [];
	recog_boxes = [];
	vehicle_boxes = [];
	num_detection = int(num[0])
	for i in range(num_detection):
		class_id = classes[0][i]
		if class_id == 2:
			x1 = int(boxes[0][i][1] * image.shape[1])
			y1 = int(boxes[0][i][0] * image.shape[0])
			x2 = int(boxes[0][i][3] * image.shape[1])
			y2 = int(boxes[0][i][2] * image.shape[0])
		# if class_id == 1:
		# 	x1 = int(boxes[0][i-1][1] * image.shape[1])
		# 	y1 = int(boxes[0][i-1][0] * image.shape[0])
		# 	x2 = int(boxes[0][i-1][3] * image.shape[1])
		# 	y2 = int(boxes[0][i-1][2] * image.shape[0])
			cropped_img = image[y1:y2, x1:x2]
			recog_result1, recog_box1 = recognition_vehicle(cropped_img)
			if len(recog_result1) > 0:
				recog_results.append(recog_result1[0])
				recog_boxes.append((x1 + recog_box1[0][0], y1 + recog_box1[0][1], x1 + recog_box1[0][2], y1 + recog_box1[0][3]))
				vehicle_boxes.append((x1, y1, x2, y2))
	return recog_results, recog_boxes, vehicle_boxes


def read_video_file(video_path: str):
    video_path = os.path.expanduser(video_path)
    cap = cv2.VideoCapture(video_path)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    return cap, video_fps

def img_infer(file_name):

	global detection_graph
	global detect_sess
	global	recog_graph
	global recog_sess
	file_bytes = np.frombuffer(file_name, np.uint8)
	image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	
	if image is None:
		print('image is null')
		sys.exit()
	image = cv2.resize(image, (1024, 640))
	detection_graph, detect_sess = load_detect_graph()
	recog_graph, recog_sess = load_recog_graph()
		
	return recognition_license_plate(image)
	
		

if __name__ == '__main__':

	global detection_graph
	global detect_sess
	global	recog_graph
	global recog_sess
	detection_graph = None
	detect_sess = None
	recog_graph = None
	recog_sess = None

	img_infer("")
	