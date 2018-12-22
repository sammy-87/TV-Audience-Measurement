import cv2
import numpy as np
from PIL import Image
from pytesseract import image_to_string
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # img = cv2.imread('test16.png',0)
    # img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    # img = cv2.medianBlur(img,5)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.imshow("Input", img)


    # img = cv2.medianBlur(img,5)

    ret,th1 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)

    # th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                # cv2.THRESH_BINARY,11,2)

    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th2, th1]
    # print(3)
    # th1 = 255 - th1
    # cv2.imshow("1", th1)
    # cv2.imwrite("test12.png",th1)
    # th2 = 255 - th2
    cv2.imshow("2", th2)
    list1 =  image_to_string(img)
    list2 =  image_to_string(th2)

    # print(list1)
    # print(len(list1))
    ''.join(list1)
    ''.join(list2)
    query = 'Main Hoon Wanted'
    print(fuzz.token_set_ratio(query, list1))
    print(fuzz.token_set_ratio(query, list2))
    # print image_to_string(img)
    # print image_to_string(th1)


    # cv2.imwrite("test13.png",th2)
    # th3 = 255 - th3
    # cv2.imshow("3", th3)

    # print(4)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # cv2.destroyAllWindows()
    # for i in xrange(4):
    #     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]),plt.yticks([])
    # plt.show()
cap.release()
cv2.destroyAllWindows()