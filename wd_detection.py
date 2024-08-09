from __future__ import division

import math
import numpy as np
import cv2
from matplotlib import pyplot as plt

import wd_preprocessing

def detect_waves(imgray,original_img):
        print("Started Detection")
        kernel=np.ones((5,5),np.uint8)
        imgCanny=cv2.Canny(imgray,100,70)
        imgDialation= cv2.dilate(imgCanny,kernel,iterations=2)
        imgEroded= cv2.erode(imgDialation,kernel,iterations=1)

        th2 = cv2.adaptiveThreshold(imgEroded,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
        
        #plt.imshow(th2,'gray')
        #plt.show()
        contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try: hierarchy = hierarchy[0]
        except: hierarchy = []
        
        x_list=[]
        y_list=[]
        xw_list=[]
        yh_list=[]
        
        # computes the bounding box for the contour, and draws it on the frame,
        found_any=False
        for contour, hier in zip(contours, hierarchy):
                (x,y,w,h) = cv2.boundingRect(contour)

                if w > 50 and w < 450 and h > 50 and h<450 and (x+w)<490:
                        found_any=True

                        x_list.append(x)
                        y_list.append(y)
                        xw_list.append((x+w))
                        yh_list.append((y+h))

                        cv2.rectangle(original_img, (x,y), (x+w,y+h), (0,255, 255), 2)
                        #cv2.drawContours(original_img, contour, -1, (0, 255, 0), 5)
        
        #print(min(x_list),min(y_list),max(xw_list),max(yh_list))
                        #cv2.putText(original_img,"Wave Found",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        if(found_any==False):
                cv2.putText(original_img,"No Wave is Detected",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

        else:
                #cv2.rectangle(original_img, (min(x_list),min(y_list)), (max(xw_list),max(yh_list)), (0,255, 0), 2)
                #cv2.putText(original_img,"Region",(min(x_list),min(y_list)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
                pass

                
        return original_img

def detect_wave_boundaries(imgray,original_img):
        kernel=np.ones((5,5),np.uint8)
        imgCanny=cv2.Canny(imgray,100,70)
        imgDialation= cv2.dilate(imgCanny,kernel,iterations=2)
        imgEroded= cv2.erode(imgDialation,kernel,iterations=1)

        th2 = cv2.adaptiveThreshold(imgEroded,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
        
        
        contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try: hierarchy = hierarchy[0]
        except: hierarchy = []
        
        x_list=[]
        y_list=[]
        xw_list=[]
        yh_list=[]

        # computes the bounding box for the contour, and draws it on the frame,
        found_any=False
        for contour, hier in zip(contours, hierarchy):
                (x,y,w,h) = cv2.boundingRect(contour)

                if w > 50 and w < 450 and h > 50 and h<450 and (x+w)<490:
                        found_any=True

                        x_list.append(x)
                        y_list.append(y)
                        xw_list.append((x+w))
                        yh_list.append((y+h))

        return (x_list,y_list,xw_list,yh_list)




    
