# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
 

video_filepath = 'test.mp4'


cap = cv2.VideoCapture(video_filepath)
ret, frame = cap.read()
count = 0

frames = []

while(True):
    ret, frame = cap.read()
    
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         
        # gray = cv2.threshold(gray, 0, 255,
        #   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
         

        # gray = cv2.medianBlur(gray, 3)
         
        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it

        text = pytesseract.image_to_string(gray)
         
        # cv2.imshow("Image", image)
        # cv2.imshow("Output", gray)
        # cv2.waitKey(0)
        
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break

        
    else: 
        break    

cap.release()
cv2.destroyAllWindows()


