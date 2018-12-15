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

frames = []

while(True):
    ret, frame = cap.read()
    
    if np.sum(frame) == 0:
        continue
    
    frames.append(frame)

    count = count + 1
    if count == 10:
        break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()


files, templates =  utils.get_templates(params.icon_dir)


array = []
for template in templates:
    

    # template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]



    for image in frames:
        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None
     
        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
     
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            res = cv2.matchTemplate(resized,template,cv2.TM_CCOEFF_NORMED) 
              
            # Specify a threshold 
            threshold = 0.8
              
            # Store the coordinates of matched area in a numpy array 
            loc = np.where( res >= threshold)  
              
            # Draw a rectangle around the matched region. 
            for pt in zip(*loc[::-1]): 
                cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 
            array.append(np.sum(res))  

import pdb; pdb.set_trace()
plt.plot(array)
plt.show()            
            



        # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)






# # Python program to illustrate  
# # template matching 
# import cv2 
# import numpy as np 
  
# # Read the main image 
# img_rgb = cv2.imread('mainimage.jpg'). 
  
# # Convert it to grayscale 
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
  
# # Read the template 
# template = cv2.imread('template',0) 
  
# # Store width and heigth of template in w and h 
# w, h = template.shape[::-1] 
  
# # Perform match operations. 
# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
  
# # Specify a threshold 
# threshold = 0.8
  
# # Store the coordinates of matched area in a numpy array 
# loc = np.where( res >= threshold)  
  
# # Draw a rectangle around the matched region. 
# for pt in zip(*loc[::-1]): 
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 
  
# # Show the final image with the matched area. 
# cv2.imshow('Detected',img_rgb)         
