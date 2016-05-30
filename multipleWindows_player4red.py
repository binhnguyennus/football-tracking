import numpy as np
import cv2
cap = cv2.VideoCapture('videofu.avi')

# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
r,h,c,w = 200,20,2248,7  # simply hardcoded the values player2 

#row, hight of the window, col, width of the window
#ddddd =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
ddddd =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
for i in range(202,215):
    for j in range(2354,2360):
        print i,"  ",j,"  ",ddddd[i,j,:]
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]


hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


mask = cv2.inRange(hsv_roi, np.array((0., 60.,64.)), np.array((18.,240.,237.)))


roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])


cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)


# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
count=0
while(cap.isOpened()):
    ret ,frame = cap.read()
    count+=1
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window

        cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

#        box = cv2.cv.BoxPoints(rect)
#        box = np.int0(box)
#        print ('x = ', x+w/2)
#        print ('y = ', y+h/2)
        # Draw it on image
#        pts = cv2.boxPoints(ret)
#        pts = np.int0(pts)
#        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.namedWindow('frame', cv2.WINDOW_OPENGL)
        cv2.imshow('frame',frame)
        k = cv2.waitKey(60) & 0xff
        if count==1:
            cv2.imwrite("fffaaaa.jpg",frame)
        if k == 27:
            break
#        else:
#            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv2.destroyAllWindows()
cap.release()