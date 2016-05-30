# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import time
import math
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
cap = cv2.VideoCapture('football_right.mp4')
ret, frame = cap.read()
img1 = cv2.imread('Soccer-mid.jpg',0)          # queryImage
img2 = cv2.imread('Soccer-right.jpg',0) # trainImage
cap = cv2.VideoCapture('football_right.mp4')
cap1 = cv2.VideoCapture('football_mid.mp4')
start = time.localtime(time.time()).tm_sec
# Initiate SURF detector
sift=cv2.SIFT()

height = frame.shape[0]
width = frame.shape[1]*2
layers = frame.shape[2]
count=0
while(cap.isOpened() and cap1.isOpened()):
    ret, frame = cap.read()
    ret1,frame1=cap1.read()
    image1=frame
    image2=frame1
    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # find the keypoints and descriptors with SURF
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
        
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)       
        
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)
        out=cv2.warpPerspective(image1,M,(image1.shape[1]+image2.shape[1],image1.shape[0]))
        for i in range (0,image2.shape[0]):
            for j in range (0,image2.shape[1]):
                if out[i,j,0]==0 and out[i,j,1]==0 and out[i,j,2]==0:
                    out[i,j,0]=image2[i,j,0]
                    out[i,j,1]=image2[i,j,1]
                    out[i,j,2]=image2[i,j,2]
                    
        count+=1
        cv2.imwrite("28.jpg",out)
        print count
        
    else:
        print "Not enough matches are found", len(good),"   ",MIN_MATCH_COUNT
        matchesMask = None
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    #cv2.namedWindow('Video', cv2.WINDOW_OPENGL)
    #cv2.imshow('Video',img3)
    tim_diff=time.localtime(time.time()).tm_sec
    tim_diff=math.fabs(tim_diff-start)
    if cv2.waitKey(1) & 0xFF == 27 or count>=1:
        break

cap.release()
cap1.release()
cv2.destroyAllWindows()
#plt.imshow(img3, 'gray'),plt.show()
#cv2.imwrite("25.jpg",img3)