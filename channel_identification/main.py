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

num_detected = []

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


def solver(img, template):
    
    method = eval(methods[4])  # cv2.TM_SQDIFF
    w, h = template.shape[::-1]
    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(np.array(img), top_left, bottom_right, 255, 2)
    print('Max value: ', max_val)
    return max_val
    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    

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


# for image in frames:
#     template = templates[1]
#     (tH, tW) = template.shape[:2]
#     found = None
#     gray  = image
#     # loop over the scales of the image
#     for scale in np.linspace(0.2, 1.0, 20)[::-1]:
#         # resize the image according to the scale, and keep track
#         # of the ratio of the resizing
#         resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
#         r = gray.shape[1] / float(resized.shape[1])
 
#         # if the resized image is smaller than the template, then break
#         # from the loop
#         if resized.shape[0] < tH or resized.shape[1] < tW:
#             break

#         # detect edges in the resized, grayscale image and apply template
#         # matching to find the template in the image
#         edged = cv2.Canny(resized, 50, 200)
#         result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
#         (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
 
#         # check to see if the iteration should be visualized
#         if args.get("visualize", False):
#             # draw a bounding box around the detected region
#             clone = np.dstack([edged, edged, edged])
#             cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
#                 (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
#             cv2.imshow("Visualize", clone)
#             cv2.waitKey(0)
 
#         # if we have found a new maximum correlation value, then update
#         # the bookkeeping variable
#         if found is None or maxVal > found[0]:
#             found = (maxVal, maxLoc, r)
#     import pdb; pdb.set_trace()
#     # unpack the bookkeeping variable and compute the (x, y) coordinates
#     # of the bounding box based on the resized ratio
#     (_, maxLoc, r) = found
#     (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
#     (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    
#     # draw a bounding box around the detected result and display the image
#     cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
#     import pdb; pdb.set_trace()
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)