import numpy as np
import cv2
from matplotlib import pyplot as plt


def display_image(img_detect):
    cv2.imshow('Wave Detector',img_detect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()