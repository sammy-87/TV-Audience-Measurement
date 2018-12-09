import cv2 
import numpy as np 
sys.path.insert(0, '../data')
import utils

video_filepath = data_dir + '/test_data/' + 'videos/' + 'test1.mp4'
frames = utils.get_frames(video_filepath)
templates =  utils.get_templates()


for i in range(len(frames)):

 
	img_rgb = frames[i]  
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
	  
	