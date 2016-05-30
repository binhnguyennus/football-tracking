import numpy as np
import cv2

cap = cv2.VideoCapture('videoxx-5min.avi')
ret, frame = cap.read()
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
if ret==True:
    Median=np.median(frame)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
cap = cv2.VideoCapture('videoxx-5min.avi')
Frame_width=int(cap.get(3))
Frame_hight=int(cap.get(4))
Frame_rate=int(cap.get(5))

fr=0.00
alpha=1.0

bgimg=cv2.imread("background.jpg")

B=np.zeros([Frame_hight,Frame_width,3])

cv2.namedWindow("normImg",cv2.WINDOW_OPENGL)
cv2.namedWindow("Frame",cv2.WINDOW_OPENGL)
H=np.matrix([[1.35145457e+01,3.19216300e+01,-2.56730457e+04],[6.29048034e-01,5.63744283e+01,-1.00889970e+04],[3.11140148e-04,3.75316123e-02,1.00000000e+00]])
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        
        objects=frame-bgimg
        
        
        objects[objects<Median]=255
        objects=np.absolute(255-objects)
        
#        objects = cv2.morphologyEx(objects, cv2.MORPH_CLOSE, kernel)
        objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, kernel)
        objects = cv2.dilate(objects,kernel,iterations = 1)
        cv2.imshow('normImg', objects)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()