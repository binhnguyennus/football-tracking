import numpy as np
import cv2

cap = cv2.VideoCapture('videofu.avi')
ret, frame = cap.read()
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
if ret==True:
    Median=np.median(frame)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
cap = cv2.VideoCapture('videofu.avi')
Frame_width=int(cap.get(3))
Frame_hight=int(cap.get(4))
Frame_rate=int(cap.get(5))

fr=0.00
alpha=1.0

bgimg=cv2.imread("background1.jpg")
img2=cv2.imread('soccer_field_1_1800x1146.png')

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
#        objects[objects<Median/2]=0
#        objects[objects>=Median/2]=255
    
#        objects=cv2.cvtColor(objects, cv2.COLOR_BGR2GRAY)
#        
#        objects[objects<Median/3]=0
#        objects[objects>=Median/3]=255
        
        
#        objects = cv2.morphologyEx(objects, cv2.MORPH_CLOSE, kernel)
        objects = cv2.morphologyEx(objects, cv2.MORPH_OPEN, kernel)
        objects = cv2.dilate(objects,kernel,iterations = 1)
        
#        objects[:,:,0]=objects[:,:,0]*4
#        objects[:,:,2]=objects[:,:,2]*4
#        objects[objects<Median]=0
        Newimage=np.zeros([img2.shape[0],img2.shape[1],3])
        Newimage=cv2.warpPerspective(objects,H,(img2.shape[1],img2.shape[0]))
        
        Newimage=Newimage+img2
        cv2.imshow('normImg', Newimage)

#        Mm=cv2.moments(objects)
#        m00=Mm['m00']
#        m01=Mm['m01']
#        m10=Mm['m10']
#        posX = m10 / m00;
#        posY = m01 / m00;
#        print posY,"     ",posX
        cv2.imshow('Frame', frame)
        #cv2.imwrite("wwwww.jpg",objects)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()