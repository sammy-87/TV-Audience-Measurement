import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
from fillholes import fillholes
import pdb
import os

# cap = cv2.VideoCapture('../data/test2.mp4')
cap = cv2.VideoCapture(1)

corners = []
colorCorners = []
edgeImages = []
avgEdgeImages = []
hystImage = []
prevAvgEdgeImages = [0,0,0,0]
offset = 0
eta_ref = 10
count = 1
area_sum = np.zeros(4)
template_dir1 = '/Users/siriusA/Documents/MyProjects/TV-Audience-Measurement/data/logo_template_1/'
template_dir2 = '/Users/siriusA/Documents/MyProjects/TV-Audience-Measurement/data/logo_template_2/'
template_dir3 = '/Users/siriusA/Documents/MyProjects/TV-Audience-Measurement/data/logo_template_3/'
method = eval('cv2.TM_CCOEFF')
big_count = 0
template_dir = template_dir1
# mask_list = [np.zeros((length, length)), np.zeros((length, length)), np.zeros((length, length)), np.zeros((length, length))]

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

while(True):
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    length_h = w//5
    length_v = h//4
    colorFrame = frame
    colorCorners.append(colorFrame[offset:length_v+offset, offset:length_h+offset])
    colorCorners.append(colorFrame[offset:length_v+offset, w-offset-length_h:w-offset])
    colorCorners.append(colorFrame[h-offset-length_v:h-offset, offset:length_h+offset])
    colorCorners.append(colorFrame[h-offset-length_v:h-offset, w-offset-length_h:w-offset])

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.uint8)
    alpha = (count-1)/count if (count <= eta_ref) else (eta_ref-1)/eta_ref
    # alpha = 1-alpha
    corners.append(image[offset:length_v+offset, offset:length_h+offset])
    corners.append(image[offset:length_v+offset, w-offset-length_h:w-offset])
    corners.append(image[h-offset-length_v:h-offset, offset:length_h+offset])
    corners.append(image[h-offset-length_v:h-offset, w-offset-length_h:w-offset])

    # finalMaxArea = 0
    # finalMaxContour = None
    for i in range(4):
        blur =cv2.GaussianBlur(corners[i], (3,3), 0)
        edgeImages.append(auto_canny(blur))
        avgEdgeImages.append(alpha*prevAvgEdgeImages[i] + (1-alpha)*edgeImages[i])
        avgEdgeImages[i] = avgEdgeImages[i].astype(np.uint8)

        hyst_threshold, _ = cv2.threshold(avgEdgeImages[i], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # hyst_threshold = 200
        temp = filters.apply_hysteresis_threshold(avgEdgeImages[i], hyst_threshold, hyst_threshold)*255
        temp = temp.astype(np.uint8)
        hystImage.append(temp)

        maxContourArea = 0
        maxContour = None
        (_, contours, _) = cv2.findContours(hystImage[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for (t, c) in enumerate(contours):
            # print("\tSize of contour %d: %d" % (t, len(c)))
            area = cv2.contourArea(c)
            if(area > maxContourArea):
                maxContourArea = area
                maxContour = c
        area_sum[i] += maxContourArea
        # print("Area of corner "+str(i), maxContourArea)
        # if (maxContourArea > finalMaxArea):
        #     finalMaxArea = maxContourArea
        #     finalMaxContour = maxContour

        mask = np.zeros((length_v, length_h, 3), dtype="uint8")

        # draw white rectangles for each object's bounding box
        (x, y, width, height) = cv2.boundingRect(maxContour)
        cv2.rectangle(mask, (x, y), (x + width, y + height), (0, 255, 0), -1)

        # mask_list[i] = mask_list[i] + (mask//255)
        # apply mask to the original image
        colorCorners[i] = cv2.bitwise_and(colorCorners[i], mask)
        # cv2.drawContours(colorCorners[i], maxContour, -1, (255, 255, 255), 5)
        cv2.imshow('frame'+str(i), colorCorners[i])
    # cv2.imshow('color image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count = count + 1

    max_res = 0
    max_channel = None
    if(count%300 == 0):
        big_count += 1
        which_corner = np.argmax(area_sum)
        which_corner = 1
        # if which_corner == 0:
        #     corner_name = 'top-left'
        # elif which_corner == 1:
        #     corner_name = 'top-right'
        # elif which_corner == 2:
        #     corner_name = 'bottom-left'
        # else:
        #     corner_name = 'bottom-right'
        # # mask_list[which_corner] = mask_list[which_corner]/300.0
        # # myvar = mask_list[which_corner] > 0.50
        # # myvar = myvar.astype(np.uint8)*255
        for filename in os.listdir(template_dir):
            if(filename == '.DS_Store'):
                continue
            filepath = template_dir+filename
            template = cv2.imread(filepath, 0)
            template_height, template_width = template.shape
            res = cv2.matchTemplate(corners[which_corner], template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + template_height, top_left[1] + template_width)

            # threshold = 1.0
            # loc = np.where( res >= threshold)
            # print(filename, np.max(res), np.min(res))
            # cv2.rectangle(img, top_left, bottom_right, 255, 2)
            if(max_val > max_res):
                max_res = np.max(res)
                max_channel = filename
        area_sum = np.zeros(4)
        # mask_list = [np.zeros((length, length)), np.zeros((length, length)), np.zeros((length, length)), np.zeros((length, length))]
        print("Channel Name = ", max_channel.split('.')[0])
        if(big_count == 6):
            template_dir = template_dir2
        elif(big_count == 10):
            template_dir = template_dir3
        elif(big_count >= 13):
            break

    corners = []
    colorCorners = []
    edgeImages = []
    prevAvgEdgeImages = avgEdgeImages
    avgEdgeImages = []
    hystImage = []

cap.release()
cv2.destroyAllWindows()