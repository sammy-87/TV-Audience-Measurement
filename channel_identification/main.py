import sys
import cv2 
import numpy as np 
sys.path.insert(0, './data')
import utils
import imutils
import params
import matplotlib.pyplot as plt
video_filepath = params.data_dir + '/test_data/' + 'videos/' + 'test.mp4'
frames = utils.get_frames(video_filepath)
files, templates =  utils.get_templates(params.icon_dir)
import pdb; pdb.set_trace()
# files = np.sort(files)
# templates = np.sort(templates)

num_detected = []

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


def solver(img, template):
    
    method = eval(methods[2])  # cv2.TM_SQDIFF
    w, h = template.shape[::-1]
    # Apply template Matching
    # import pdb; pdb.set_trace()
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(np.array(img), top_left, bottom_right, 255, 5)
    # plt.subplot(121),plt.imshow(res, cmap='gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img, cmap='gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    return max_val


def non_causal():

    max_val = np.zeros((len(templates), len(frames), 3))
    image_count = 0

    for image in frames:
        print(" Searching for logo in video ", video_filepath)
        template_count = 0
        
        for template in templates:
            print("Logo: ", files[template_count])
            
            for channel_iter in range(image.shape[2]):
                max_val[template_count, image_count, channel_iter] = solver(image[:, :, channel_iter], template[:, :, channel_iter])
            template_count = template_count + 1
        image_count = image_count + 1

    for j in range(3):
        plt.figure()
        for i in range(len(templates)):
            plt.plot(max_val[i, :, j],  label=files[i])
            plt.legend(files)    

    plt.show()


# def corner_patches(image):
#     percentage = 0.2
#     h = 
#     corner1 = image[0:]


# def causal():
    
# cap = cv2.VideoCapture(1)

# while(True):

#     ret, frame = cap.read()
#     # image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     image = frame 
#     # print(np.sum(frame))
#     # cv2.imshow('frame',image)
#     # if cv2.waitKey(0) & 0xFF == ord('q'):
#     #     break
#     if ret == True:
#         max_val = []    
#         template_count = 0
#         for template in templates:
#             # print("Testing for Logo: ", files[template_count])
#             channel_val = []
#             for channel_iter in range(image.shape[2]):
#                 channel_val.append(solver(image[:, :, channel_iter], template[:, :, channel_iter])) 
            
#             # import pdb; pdb.set_trace()
#             # print (files[template_count], np.mean(template))
#             # print("Shape of template: ", template.shape, "NP average ")
#             max_val.append(np.max(channel_val))

#             template_count = template_count + 1
#             # import pdb; pdb.set_trace()
        
#         # import pdb; pdb.set_trace()
#         arg_max = np.argmax(max_val)
#         detected_logo = files[arg_max]
        
#         print("Detected Channel Logo is ::", detected_logo)
#     else:
#         break
# # causal()


non_causal()