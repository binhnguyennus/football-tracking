import numpy as np
import cv2
cap = cv2.VideoCapture('videofu.avi')

# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
r,h,c,w = 360,40,2310,15  # simply hardcoded the values
r1,h1,c1,w1 = 231,20,2281,10  # simply hardcoded the values
r2,h2,c2,w2 = 224,20,2078,10  # simply hardcoded the values
r3,h3,c3,w3 = 325,20,2640,10  # simply hardcoded the values

#row, hight of the window, col, width of the window
#ddddd =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
for i in range(r,r+h):
    for j in range(c,c+w):
        print i,"  ",j,"  ",frame[i,j,:]
track_window = (c,r,w,h)
track_window1 = (c1,r1,w1,h1)
track_window2 = (c2,r2,w2,h2)
track_window3 = (c3,r3,w3,h3)
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
roi1 = frame[r1:r1+h1, c1:c1+w1]
roi2 = frame[r2:r2+h2, c2:c2+w2]
roi3 = frame[r3:r3+h3, c3:c3+w3]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hsv_roi1 =  cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
hsv_roi2 =  cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
hsv_roi3 =  cv2.cvtColor(roi3, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 180.,130.)), np.array((8.,255.,200.)))
mask1 = cv2.inRange(hsv_roi1, np.array((0., 180.,130.)), np.array((8.,255.,200.)))
mask2 = cv2.inRange(hsv_roi2, np.array((0., 180.,130.)), np.array((8.,255.,200.)))
mask3 = cv2.inRange(hsv_roi3, np.array((0., 180.,130.)), np.array((8.,255.,200.)))

roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
roi_hist1 = cv2.calcHist([hsv_roi1],[0],mask1,[180],[0,180])
roi_hist2 = cv2.calcHist([hsv_roi2],[0],mask2,[180],[0,180])
roi_hist3 = cv2.calcHist([hsv_roi3],[0],mask3,[180],[0,180])

cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist1,roi_hist1,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist2,roi_hist2,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist3,roi_hist3,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
count=0
while(cap.isOpened()):
    ret ,frame = cap.read()
    count+=1
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        dst1 = cv2.calcBackProject([hsv],[0],roi_hist1,[0,180],1)
        dst2 = cv2.calcBackProject([hsv],[0],roi_hist2,[0,180],1)
        dst3 = cv2.calcBackProject([hsv],[0],roi_hist3,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        ret1, track_window1 = cv2.meanShift(dst1, track_window1, term_crit)
        ret2, track_window2 = cv2.meanShift(dst2, track_window2, term_crit)
        ret3, track_window3 = cv2.meanShift(dst3, track_window3, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        x1,y1,w1,h1 = track_window1
        x2,y2,w2,h2 = track_window2
        x3,y3,w3,h3 = track_window3
        cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.rectangle(frame, (x1,y1), (x1+w1,y1+h1), 255,2)
        cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), 255,2)
        cv2.rectangle(frame, (x3,y3), (x3+w3,y3+h3), 255,2)
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