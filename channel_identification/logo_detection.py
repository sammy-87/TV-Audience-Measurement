import cv2
import numpy as np
from skimage import filters
from fillholes import fillholes
import pdb

cap = cv2.VideoCapture('data/test3.mp4')

corners = []
edgeImages = []
avgEdgeImages = []
hystImage = []
prevAvgEdgeImages = [0,0,0,0]
length = 200
eta_ref = 20
low = 30
high = 100
count = 1

while(True):
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(image.shape)              # 480 x 720
    kernel = np.ones((5,5),np.uint8)
    alpha = (count-1)/count if (count <= eta_ref) else (eta_ref-1)/eta_ref
    corners.append(image[0:length, 0:length])
    corners.append(image[480-length:480, 0:length])
    corners.append(image[0:length, 720-length:720])
    corners.append(image[480-length:480, 720-length:720])

    for i in range(4):
        edgeImages.append(cv2.Canny(corners[i], 30, 100))
        avgEdgeImages.append(alpha*prevAvgEdgeImages[i] + (1-alpha)*edgeImages[i])
        temp = filters.sobel(avgEdgeImages[i])

        hystImage.append(filters.apply_hysteresis_threshold(temp, low, high))
        hystImage[i] = hystImage[i].astype(np.uint8)*255
        # pdb.set_trace()
        closing = cv2.morphologyEx(hystImage[i], cv2.MORPH_CLOSE, kernel)
        holesFilled = fillholes(closing)
        opening = cv2.morphologyEx(holesFilled, cv2.MORPH_OPEN, kernel)

        # ret, labels = cv2.connectedComponents(opening)
        # pdb.set_trace()
        cv2.imshow('frame'+str(i), opening)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    corners = []
    edgeImages = []
    prevAvgEdgeImages = avgEdgeImages
    avgEdgeImages = []
    hystImage = []
    count = count + 1

cap.release()
cv2.destroyAllWindows()