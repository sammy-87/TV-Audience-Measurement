import cv2
import numpy as np
import os

def get_num_frames(filepath):
	cap = cv2.VideoCapture(filepath)
	print ("Reading file: ", filepath)
	count = 0
	while(cap.isOpened() == True):
		ret, frame = cap.read()	
		if ret == True:
			count = count + 1
		else: 
			break
	return count		

def get_frames(filepath):

	cap = cv2.VideoCapture(filepath)
	print ("Reading file: ", filepath)
	output_frames = []
	num_frames = get_frames(filepath)
	print("Number of frames = ", num_frames)
	frame_index  = np.linspace(0, num_frames, num=10)
	frame_index = int(frame_index)
	count = 0
	while(cap.isOpened() == True):
		ret, frame = cap.read()
		if ret == True:
			count = count + 1
			if count in frame_index:
				output_frames.append(frame)

		else: 
			break
	return output_frames


def get_templates():
	data_dir = 'test_data/icons'
	files = os.listdir(data_dir) 
	templates = []
	for file in len(files):
		if file.split('.')[-1] != '.png' or '.jpg':
			template = cv2.imread(data_dir + '/' + file ,0) 
			template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
			template = cv2.Canny(template, 50, 200)
			templates.append(template)
