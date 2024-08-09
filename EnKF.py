from filterpy.kalman import EnsembleKalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
import cv2
from matplotlib import pyplot as plt
import wd_detection
import wd_preprocessing


def hx(x):
    return np.array([x[0]])

F = np.array([[1., 1.],
              [0., 1.]])
def fx(x, dt):
    return np.dot(F, x)


x = np.array([0., 1.])
P = np.eye(2) * 100.
dt = 0.1
f = EnsembleKalmanFilter(x=x, P=P, dim_z=1, dt=dt, N=8,
            hx=hx, fx=fx)

std_noise = 3.
f.R *= std_noise**2
f.Q = Q_discrete_white_noise(2, dt, .01)

loc= input("Enter the name of the image file along with extension : ")
im = cv2.imread(loc)
img = wd_preprocessing.resize_img(im)
im_p = wd_preprocessing.preprocess(img)
#img_detect = wd_detection.detect_waves(im_p,img)

x_list=wd_detection.detect_wave_boundaries(im_p,img)[0]
y_list=wd_detection.detect_wave_boundaries(im_p,img)[1]
xw_list=wd_detection.detect_wave_boundaries(im_p,img)[2]
yh_list=wd_detection.detect_wave_boundaries(im_p,img)[3]

x_up=[]
y_up=[]
xw_up =[]
yh_up=[]
for i in range(len(x_list)):
    z=x_list[i]
    f.predict()
    x_up.append(f)
    f.update(np.asarray([z]))
    z=y_list[i]
    f.predict()
    y_up.append(f)
    f.update(np.asarray([z]))
    z=xw_list[i]
    f.predict()
    xw_up.append(f)
    f.update(np.asarray([z]))
    z=yh_list[i]
    f.predict()
    yh_up.append(f)
    f.update(np.asarray([z]))

    cv2.rectangle(img, (x_list[i],y_list[i]), (xw_list[i],yh_list[i]), (0,255, 255), 2)
cv2.imshow('Wave Detector',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

