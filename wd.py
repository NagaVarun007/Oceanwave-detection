import numpy as np
import cv2
from matplotlib import pyplot as plt

import wd_detection
import wd_preprocessing
import wd_io
import EnKF

def analyze(loc):
    im = cv2.imread(loc)
    img = wd_preprocessing.resize_img(im)
    im_p = wd_preprocessing.preprocess(img)
    #img_detect = wd_detection.detect_waves(im_p,img)
    bou=wd_detection.detect_wave_boundaries(im_p,img)
    x_list=bou[0]
    y_list=bou[1]
    xw_list=bou[2]
    yh_list=bou[3]
    print(x_list)
    #EnKF.apply(im,x_list,y_list,xw_list,yh_list)

    #wd_io.display_image(img_detect)


def main():
    inputfile = input("Enter the name of the image file along with extension : ")
    print("Checking the image ")
    analyze(inputfile)

main()