#!/usr/bin/python

# 2.12 Lab 7 object detection: a node for color thresholding
# Jacob Guggenheim 2019
# Jerry Ng 2019

import rospy
import numpy as np
import cv2  # OpenCV module
from matplotlib import pyplot as plt
import time
from Tkinter import *
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Twist, Vector3, Quaternion
from std_msgs.msg import ColorRGBA

from cv_bridge import CvBridge, CvBridgeError
import message_filters
import math

rospy.init_node('colorThresh', anonymous=True)

# Bridge to convert ROS Image type to OpenCV Image type
cv_bridge = CvBridge()
tk = Tk()
l_b = Scale(tk, from_ = 0, to = 255, label = 'Blue, lower', orient = HORIZONTAL)
l_b.pack()
u_b = Scale(tk, from_ = 0, to = 255, label = 'Blue, upper', orient = HORIZONTAL)
u_b.pack()
u_b.set(255)
l_g = Scale(tk, from_ = 0, to = 255, label = 'Green, lower', orient = HORIZONTAL)
l_g.pack()
u_g = Scale(tk, from_ = 0, to = 255, label = 'Green, upper', orient = HORIZONTAL)
u_g.pack()
u_g.set(255)
l_r = Scale(tk, from_ = 0, to = 255, label = 'Red, lower', orient = HORIZONTAL)
l_r.pack()
u_r = Scale(tk, from_ = 0, to = 255, label = 'Red, upper', orient = HORIZONTAL)
u_r.pack()
u_r.set(255)

l_h = Scale(tk, from_ = 0, to = 255, label = 'Hue, lower', orient = HORIZONTAL)
l_h.pack()
u_h = Scale(tk, from_ = 0, to = 255, label = 'Hue, upper', orient = HORIZONTAL)
u_h.pack()
u_h.set(255)
l_s = Scale(tk, from_ = 0, to = 255, label = 'Saturation, lower', orient = HORIZONTAL)
l_s.pack()
u_s = Scale(tk, from_ = 0, to = 255, label = 'Saturation, upper', orient = HORIZONTAL)
u_s.pack()
u_s.set(255)
l_v = Scale(tk, from_ = 0, to = 255, label = 'Value, lower', orient = HORIZONTAL)
l_v.pack()
u_v = Scale(tk, from_ = 0, to = 255, label = 'Value, upper', orient = HORIZONTAL)
u_v.pack()
u_v.set(255)


def main():
    rospy.Subscriber('/camera/rgb/image_raw', Image, colorThreshCallback)
    print("Subscribing")
    mainloop()


def colorThreshCallback(msg):
    # convert ROS image to opencv format

    print('here')
    i = 0
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    # visualize it in a cv window
    cv2.imshow("Original_Image", cv_image)
    cv2.waitKey(3)
    ################ RGB THRESHOLDING ####################
    #get threshold values
    
    lower_bound_RGB = np.array([l_b.get(), l_g.get(), l_r.get()])
    upper_bound_RGB = np.array([u_b.get(), u_g.get(), u_r.get()])

    # threshold
    mask_RGB = cv2.inRange(cv_image, lower_bound_RGB, upper_bound_RGB)

    # get display image
    disp_image_RGB = cv2.bitwise_and(cv_image,cv_image, mask= mask_RGB)
    cv2.imshow("RGB_Thresholding", disp_image_RGB)
    cv2.waitKey(3)


    ################ HSV THRESHOLDING ####################
    # conver to HSV
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # get threshold values
    lower_bound_HSV = np.array([l_h.get(), l_s.get(), l_v.get()])
    upper_bound_HSV = np.array([u_h.get(), u_s.get(), u_v.get()])

    # threshold
    mask_HSV = cv2.inRange(hsv_image, lower_bound_HSV, upper_bound_HSV)

    # get display image
    disp_image_HSV = cv2.bitwise_and(cv_image,cv_image, mask= mask_HSV)
    cv2.imshow("HSV_Thresholding", disp_image_HSV)
    cv2.waitKey(3)

if __name__=='__main__':
    main()
