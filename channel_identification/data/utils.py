import cv2
import numpy as np
import os

def get_num_frames(filepath):
    cap = cv2.VideoCapture(filepath)
    print ("Counting number of frames : ", filepath)
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
    window_length = 100
    num_frames = 10
    output_frames = []
    print("Number of frames = ", num_frames)
    # num_frames = get_num_frames(filepath)
    # frame_index  = np.linspace(0, num_frames, num=100)
    # frame_index = frame_index.astype(int)
    count = 0
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret == True:            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output_frames.append(frame)
        else: 
            break

    return output_frames


def get_templates(icon_dir):

    files = os.listdir(icon_dir) 
    templates = []
    for file in files:
        if file.split('.')[-1] != '.png' or '.jpg':
            template = cv2.imread(icon_dir + '/' + file)
            # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            # template = cv2.Canny(template, 50, 200)
            templates.append(template)        
    return files, templates
