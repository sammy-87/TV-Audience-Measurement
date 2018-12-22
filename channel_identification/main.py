import sys
import cv2 
import numpy as np 
sys.path.insert(0, './data')
import utils
import imutils
import params
import matplotlib.pyplot as plt
video_filepath = params.data_dir + '/test_data/' + 'videos/' + 'test.mp4'
cap = cv2.VideoCapture(video_filepath)

count = 0

images = []

while(True):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.sum(frame) == 0:
        continue
    
    else:
        images.append(frame)

        count = count + 1
        if count == 10:
            break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()

def get_corners(img):
    corners = []
    length = 300
    height = img.shape[0]
    width = img.shape[1]
    corners.append(img[0:length, 0:length])
    corners.append(img[height-length:height, 0:length])
    corners.append(img[0:length, width-length:width])
    corners.append(img[height-length:height, width-length:width])

    return corners
frames = []

for image in images:
    frames.append(get_corners(image))

files, templates =  utils.get_templates(params.icon_dir)


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

    max_val = []
    
    for corners in frames:
        print(" Searching for logo in video ", video_filepath)
        template_count = 0
        
        for template in templates:
            val = []
            for corner in corners:
                if np.sum(corner) == 0:
                    continue
                val.append(solver(corner, template))
            template_count = template_count + 1
            max_val.append(max(val))
        import pdb; pdb.set_trace()    



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