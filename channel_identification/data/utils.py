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
        
        if np.sum(frame) == 0:
            continue
        
        if ret == True:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output_frames.append(frame)
        else: 
            break

    return output_frames


def get_templates(icon_dir):

    dummy_files = os.listdir(icon_dir) 
    
    files = []
    for i in range(len(dummy_files)):
        if dummy_files[i].split('.')[-1] == 'png':
            files.append(dummy_files[i])
        if dummy_files[i].split('.')[-1] == 'jpg':
            files.append(dummy_files[i])

    templates = []
    for file in files:
        template = cv2.imread(icon_dir + '/' + file)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        templates.append(template) 

    return files, templates
