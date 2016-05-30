import numpy as np
import cv2
cap = cv2.VideoCapture('videofu.avi')

# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
r,h,c,w = 169,10,2384,10  # simply hardcoded the values player8
r1,h1,c1,w1 = 200,20,2248,7  # simply hardcoded the values player4
r2,h2,c2,w2 = 170,8,2137,5  # simply hardcoded the values player3 

r3,h3,c3,w3 = 360,40,2310,15  # simply hardcoded the values player#6
r4,h4,c4,w4 = 231,20,2281,10  # simply hardcoded the values player#5
r5,h5,c5,w5 = 224,20,2078,10  # simply hardcoded the values player#2
r6,h6,c6,w6 = 312,25,2640,15  # simply hardcoded the values player#9
r7,h7,c7,w7 = 203,20,2352,10  # simply hardcoded the values player#7
r8,h8,c8,w8 = 261,10,1652,12  # simply hardcoded the values player#1

#row, hight of the window, col, width of the window
#ddddd =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
ddddd =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
for i in range(169,185):
    for j in range(2384,2390):
        print i,"  ",j,"  ",ddddd[i,j,:]
track_window = (c,r,w,h)
track_window1 = (c1,r1,w1,h1)
track_window2 = (c2,r2,w2,h2)

track_window3 = (c3,r3,w3,h3)
track_window4 = (c4,r4,w4,h4)
track_window5 = (c5,r5,w5,h5)
track_window6 = (c6,r6,w6,h6)
track_window7 = (c7,r7,w7,h7)
track_window8 = (c8,r8,w8,h8)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
roi1 = frame[r1:r1+h1, c1:c1+w1]
roi2 = frame[r2:r2+h2, c2:c2+w2]

roi3 = frame[r3:r3+h3, c3:c3+w3]
roi4 = frame[r4:r4+h4, c4:c4+w4]
roi5 = frame[r5:r5+h5, c5:c5+w5]
roi6 = frame[r6:r6+h6, c6:c6+w6]
roi7 = frame[r7:r7+h7, c7:c7+w7]
roi8 = frame[r8:r8+h8, c8:c8+w8]

hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hsv_roi1 =  cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
hsv_roi2 =  cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

hsv_roi3 =  cv2.cvtColor(roi3, cv2.COLOR_BGR2HSV)
hsv_roi4 =  cv2.cvtColor(roi4, cv2.COLOR_BGR2HSV)
hsv_roi5 =  cv2.cvtColor(roi5, cv2.COLOR_BGR2HSV)
hsv_roi6 =  cv2.cvtColor(roi6, cv2.COLOR_BGR2HSV)
hsv_roi7 =  cv2.cvtColor(roi7, cv2.COLOR_BGR2HSV)
hsv_roi8 =  cv2.cvtColor(roi8, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv_roi, np.array((0., 40.,70.)), np.array((18.,200.,150.)))
mask1 = cv2.inRange(hsv_roi1, np.array((0., 60.,64.)), np.array((18.,240.,237.)))
mask2 = cv2.inRange(hsv_roi2, np.array((5., 93.,74.)), np.array((35.,195.,137.)))

mask3 = cv2.inRange(hsv_roi3, np.array((0., 180.,130.)), np.array((8.,255.,200.)))
mask4 = cv2.inRange(hsv_roi4, np.array((0., 180.,130.)), np.array((8.,255.,200.)))
mask5 = cv2.inRange(hsv_roi5, np.array((0., 180.,130.)), np.array((8.,255.,200.)))
mask6 = cv2.inRange(hsv_roi6, np.array((0., 180.,130.)), np.array((8.,255.,200.)))#((30., 50.,140.)), np.array((172.,200.,255.)))
mask7 = cv2.inRange(hsv_roi7, np.array((0., 100.,100.)), np.array((18.,200.,200.)))
mask8 = cv2.inRange(hsv_roi8, np.array((22., 48.,28.)), np.array((34.,255.,155.)))


roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
roi_hist1 = cv2.calcHist([hsv_roi1],[0],mask1,[180],[0,180])
roi_hist2 = cv2.calcHist([hsv_roi2],[0],mask2,[180],[0,180])

roi_hist3 = cv2.calcHist([hsv_roi3],[0],mask3,[180],[0,180])
roi_hist4 = cv2.calcHist([hsv_roi4],[0],mask4,[180],[0,180])
roi_hist5 = cv2.calcHist([hsv_roi5],[0],mask5,[180],[0,180])
roi_hist6 = cv2.calcHist([hsv_roi6],[0],mask6,[180],[0,180])
roi_hist7 = cv2.calcHist([hsv_roi7],[0],mask7,[180],[0,180])
roi_hist8 = cv2.calcHist([hsv_roi8],[0],mask8,[180],[0,180])

cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist1,roi_hist1,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist2,roi_hist2,0,255,cv2.NORM_MINMAX)

cv2.normalize(roi_hist3,roi_hist3,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist4,roi_hist4,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist5,roi_hist5,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist6,roi_hist6,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist7,roi_hist7,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist8,roi_hist8,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1 )
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
        dst4 = cv2.calcBackProject([hsv],[0],roi_hist4,[0,180],1)
        dst5 = cv2.calcBackProject([hsv],[0],roi_hist5,[0,180],1)
        dst6 = cv2.calcBackProject([hsv],[0],roi_hist6,[0,180],1)
        dst7 = cv2.calcBackProject([hsv],[0],roi_hist7,[0,180],1)
        dst8 = cv2.calcBackProject([hsv],[0],roi_hist8,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        ret1, track_window1 = cv2.meanShift(dst1, track_window1, term_crit)
        ret2, track_window2 = cv2.meanShift(dst2, track_window2, term_crit)

        ret3, track_window3 = cv2.meanShift(dst3, track_window3, term_crit)
        ret4, track_window4 = cv2.meanShift(dst4, track_window4, term_crit)
        ret5, track_window5 = cv2.meanShift(dst5, track_window5, term_crit)
        ret6, track_window6 = cv2.meanShift(dst6, track_window6, term_crit)
        ret7, track_window7 = cv2.meanShift(dst7, track_window7, term_crit)
        ret8, track_window8 = cv2.meanShift(dst8, track_window8, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        x1,y1,w1,h1 = track_window1
        x2,y2,w2,h2 = track_window2

        x3,y3,w3,h3 = track_window3
        x4,y4,w4,h4 = track_window4
        x5,y5,w5,h5 = track_window5
        x6,y6,w6,h6 = track_window6
        x7,y7,w7,h7 = track_window7
        x8,y8,w8,h8 = track_window8

        cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.putText(frame,"8",(x,y-2),cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
        with open('.\\player8.txt', 'a') as f:
            f.write('x = %-15s y = %-15s\n' % (x+w/2,y+h/2))        

        cv2.rectangle(frame, (x1,y1), (x1+w1,y1+h1), 255,2)
        cv2.putText(frame,"4",(x1,y1-2),cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
        with open('.\\player4.txt', 'a') as f:
            f.write('x = %-15s y = %-15s\n' % (x1+w1/2,y1+h1/2))        

        cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), 255,2)
        cv2.putText(frame,"3",(x2,y2-2),cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
        with open('.\\player3.txt', 'a') as f:
            f.write('x = %-15s y = %-15s\n' % (x2+w2/2,y2+h2/2))        

        cv2.rectangle(frame, (x3,y3), (x3+w3,y3+h3), 255,2)
        cv2.putText(frame,"6",(x3,y3-2),cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
        with open('.\\player6.txt', 'a') as f:
            f.write('x = %-15s y = %-15s\n' % (x3+w3/2,y3+h3/2))        

        cv2.rectangle(frame, (x4,y4), (x4+w4,y4+h4), 255,2)
        cv2.putText(frame,"5",(x4,y4-2),cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
        with open('.\\player5.txt', 'a') as f:
            f.write('x = %-15s y = %-15s\n' % (x4+w4/2,y4+h4/2))        

        cv2.rectangle(frame, (x5,y5), (x5+w5,y5+h5), 255,2)
        cv2.putText(frame,"2",(x5,y5-2),cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
        with open('.\\player2.txt', 'a') as f:
            f.write('x = %-15s y = %-15s\n' % (x5+w5/2,y5+h5/2))        

        cv2.rectangle(frame, (x6,y6), (x6+w6,y6+h6), 255,2)
        cv2.putText(frame,"9",(x6,y6-2),cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
        with open('.\\player9.txt', 'a') as f:
            f.write('x = %-15s y = %-15s\n' % (x6+w6/2,y6+h6/2))        

        cv2.rectangle(frame, (x7,y7), (x7+w7,y7+h7), 255,2)
        cv2.putText(frame,"7",(x7,y7-2),cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
        with open('.\\player7.txt', 'a') as f:
            f.write('x = %-15s y = %-15s\n' % (x7+w7/2,y7+h7/2))        

        cv2.rectangle(frame, (x8,y8), (x8+w8,y8+h8), 255,2)
        cv2.putText(frame,"1",(x8,y8-2),cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
        with open('.\\player1.txt', 'a') as f:
            f.write('x = %-15s y = %-15s\n' % (x8+w8/2,y8+h8/2))        
                
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
            print 'Frame#', count
            break
#        else:
#            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv2.destroyAllWindows()
cap.release()