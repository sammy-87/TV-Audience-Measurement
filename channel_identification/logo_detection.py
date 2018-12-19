import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
from fillholes import fillholes
import pdb

cap = cv2.VideoCapture('../data/test2.mp4')

corners = []
colorCorners = []
edgeImages = []
avgEdgeImages = []
hystImage = []
prevAvgEdgeImages = [0,0,0,0]
length = 150
offset = 0
eta_ref = 10
count = 1

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
    colorFrame = frame
    colorCorners.append(colorFrame[offset:length+offset, offset:length+offset])
    colorCorners.append(colorFrame[offset:length+offset, w-offset-length:w-offset])
    colorCorners.append(colorFrame[h-offset-length:h-offset, offset:length+offset])
    colorCorners.append(colorFrame[h-offset-length:h-offset, w-offset-length:w-offset])

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       # 480 x 720
    kernel = np.ones((3,3),np.uint8)
    alpha = (count-1)/count if (count <= eta_ref) else (eta_ref-1)/eta_ref
    # alpha = 1-alpha
    corners.append(image[offset:length+offset, offset:length+offset])
    corners.append(image[offset:length+offset, w-offset-length:w-offset])
    corners.append(image[h-offset-length:h-offset, offset:length+offset])
    corners.append(image[h-offset-length:h-offset, w-offset-length:w-offset])

    # finalMaxArea = 0
    # finalMaxContour = None
    for i in range(4):
        blur =cv2.GaussianBlur(corners[i], (3,3), 0)
        edgeImages.append(auto_canny(blur))
        avgEdgeImages.append(alpha*prevAvgEdgeImages[i] + (1-alpha)*edgeImages[i])
        avgEdgeImages[i] = avgEdgeImages[i].astype(np.uint8)

        hyst_threshold, _ = cv2.threshold(avgEdgeImages[i], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # hyst_threshold  = 200
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
        # print("Area of corner "+str(i), maxContourArea)
        # if (maxContourArea > finalMaxArea):
        #     finalMaxArea = maxContourArea
        #     finalMaxContour = maxContour

        cv2.drawContours(colorCorners[i], maxContour, -1, (255, 255, 255), 5)
        # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilation, connectivity=4)
        # sizes = stats[:, -1]
        # max_label = 1
        # max_size = sizes[0]
        # for i in range(2, nb_components):
        #     if sizes[i] > max_size:
        #         max_label = i
        #         max_size = sizes[i]

        # closing = cv2.morphologyEx(hystImage[i], cv2.MORPH_CLOSE, kernel)
        # closing = closing.astype(np.uint8)
        # holesFilled = fillholes(closing)
        # opening = cv2.morphologyEx(holesFilled, cv2.MORPH_OPEN, kernel)

        # # ret, labels = cv2.connectedComponents(opening)

        # fig.add_subplot(2,2,i+1)
        # plt.imshow(edgeImages[i], cmap='gray')
        cv2.imshow('frame'+str(i), colorCorners[i])
    # cv2.drawContours(frame, finalMaxContour, -1, (255, 255, 255), 5)
    cv2.imshow('color image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    corners = []
    colorCorners = []
    edgeImages = []
    prevAvgEdgeImages = avgEdgeImages
    avgEdgeImages = []
    hystImage = []
    count = count + 1
    # if(count > 300):
    #     count = 1
    # plt.show(block=False)
    # plt.draw()
    # plt.pause(1e-12)

cap.release()
cv2.destroyAllWindows()