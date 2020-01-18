'''import cv2
import numpy as np
#from gluoncv import model_zoo, data, utils

cap = cv2.VideoCapture('vid_1.mp4')
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

ret, frame = cap.read()

#x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
#class_IDs, scores, bounding_boxs = detector(x)

cv2.imwrite("vid_1_f_1.jpg",frame)

'''

import cv2
import numpy as np

cap = cv2.VideoCapture('vid_1.mp4')
 

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.imwrite("vid_1_s_1.jpg",frame)
        break
  else: 
    break
 
cap.release()
cv2.destroyAllWindows()