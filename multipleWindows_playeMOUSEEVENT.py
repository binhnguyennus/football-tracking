import numpy as np
import cv2


from time import time
boxes = []
def on_mouse(event, x, y, flags, params):
    # global img
    t = time()

    if event == cv2.cv.CV_EVENT_LBUTTONDBLCLK:
         print 'Start Mouse Position: '+str(x)+', '+str(y)
         sbox = [x, y]
         boxes.append(sbox)
         # print count
         # print sbox

    elif event == cv2.cv.CV_EVENT_MBUTTONDBLCLK:
        print 'End Mouse Position: '+str(x)+', '+str(y)
        ebox = [x, y]
        boxes.append(ebox)
        print boxes
        crop = img[boxes[-2][1]:boxes[-1][1],boxes[-2][0]:boxes[-1][0]]

        cv2.imshow('crop',crop)
        k =  cv2.waitKey(0)
        if ord('r')== k:
            cv2.imwrite('Crop'+str(t)+'.jpg',crop)
            print "Written to file"

count = 0
while(1):
    count += 1
    img = cv2.imread('FIRST_FRAME.PNG',1)
    # img = cv2.blur(img, (3,3))
#    img = cv2.resize(img, None, fx = 0.25,fy = 0.25)

    cv2.namedWindow('real image')
    cv2.cv.SetMouseCallback('real image', on_mouse, 0)
    cv2.imshow('real image', img)
    if count < 50:
        if cv2.waitKey(33) == 27:
            cv2.destroyAllWindows()
            break
    elif count >= 50:
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break
        count = 0

cap = cv2.VideoCapture('videofu.avi')

# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
r,h,c,w = boxes[0][1],boxes[1][1]-boxes[0][1],boxes[0][0],boxes[1][0]-boxes[0][0] # simply hardcoded the values player2 

#row, hight of the window, col, width of the window
#ddddd =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
ddddd =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


for i in range(boxes[0][1],boxes[1][1]):
    for j in range(boxes[0][0],boxes[1][0]):
        print i,"  ",j,"  ",ddddd[i,j,:]
        
meann= np.mean(ddddd[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0],0])        

stddeviation = np.std(ddddd[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0],0])

meann2= np.mean(ddddd[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0],1])        

stddeviation2 = np.std(ddddd[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0],1])

meann3= np.mean(ddddd[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0],2])        

stddeviation3 = np.std(ddddd[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0],2])


track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]


hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#asdroi = frame[r:r+h, c:c+w]
#asdsd_roi =  cv2.cvtColor(asdroi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((meann-stddeviation, meann2-stddeviation2,meann3-stddeviation3)), np.array((meann+stddeviation,meann2+stddeviation2,meann3+stddeviation3)))

roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])


cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)


# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1 )
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