#!/usr/bin/python

# 2.12 Lab 4 object detection: a node for de-noising
# Luke Roberto Oct 2017

import rospy
import numpy as np
import cv2  # OpenCV module
from matplotlib import pyplot as plt
import time

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Twist, Vector3, Quaternion
from std_msgs.msg import ColorRGBA

from cv_bridge import CvBridge, CvBridgeError
import message_filters
import math

rospy.init_node('morphOps', anonymous=True)

# Bridge to convert ROS Image type to OpenCV Image type
cv_bridge = CvBridge()

def nothing(x):
    pass

def main():
    # create HSV tracker bar
    cv2.namedWindow('HSV_Thresholding')
    cv2.createTrackbar('L - h', 'HSV_Thresholding', 0, 255, nothing)
    cv2.createTrackbar('U - h', 'HSV_Thresholding', 255, 255, nothing)
    cv2.createTrackbar('L - s', 'HSV_Thresholding', 0, 255, nothing)
    cv2.createTrackbar('U - s', 'HSV_Thresholding', 255, 255, nothing)
    cv2.createTrackbar('L - v', 'HSV_Thresholding', 0, 255, nothing)
    cv2.createTrackbar('U - v', 'HSV_Thresholding', 255, 255, nothing)

    rospy.Subscriber('/usb_cam/image_raw', Image, morphOpsCallback)
    print("Subscribing")
    rospy.spin()


def morphOpsCallback(msg):
    # convert ROS image to opencv format
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)

    # visualize it in a cv window
    cv2.imshow("Original_Image", cv_image)
    cv2.waitKey(3)

    ################ HSV THRESHOLDING ####################
    # conver to HSV
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # get threshold values
    l_h = cv2.getTrackbarPos('L - h', 'HSV_Thresholding')
    u_h = cv2.getTrackbarPos('U - h', 'HSV_Thresholding')
    l_s = cv2.getTrackbarPos('L - s', 'HSV_Thresholding')
    u_s = cv2.getTrackbarPos('U - s', 'HSV_Thresholding')
    l_v = cv2.getTrackbarPos('L - v', 'HSV_Thresholding')
    u_v = cv2.getTrackbarPos('U - v', 'HSV_Thresholding')
    lower_bound_HSV = np.array([l_h, l_s, l_v])
    upper_bound_HSV = np.array([u_h, u_s, u_v])

    # threshold
    mask_HSV = cv2.inRange(hsv_image, lower_bound_HSV, upper_bound_HSV)

    # display image
    cv2.imshow("HSV_Thresholding", mask_HSV)
    cv2.waitKey(3)

    # kernel for all morphological operations
    kernel = np.ones((7,7),np.uint8)
    ################ Erosion ####################
    # erode blobs
    erosion = cv2.erode(mask_HSV,kernel,iterations = 3)

    # display image
    cv2.imshow("Erosion", erosion)
    cv2.waitKey(3)

    ################ Dilation ####################
    # dilate blobs
    dilation = cv2.dilate(mask_HSV,kernel,iterations = 3)

    # display image
    cv2.imshow("Dilation", dilation)
    cv2.waitKey(3)

    ################ Opening ####################
    # good for removing noise. its an erosion (to get rid of noise) followed by a dilation (to get back the original blobs you wanted to keep)
    opening = cv2.morphologyEx(mask_HSV, cv2.MORPH_OPEN, kernel, iterations = 3)

    # display image
    cv2.imshow("Opening - Get rid of noise", opening)
    cv2.waitKey(3)

    ################ Closing ####################
    # good for filling small holes in blobs. its a dilation (to fill the holes) followed by an erosion (to get the object back to the right size)
    closing = cv2.morphologyEx(mask_HSV, cv2.MORPH_CLOSE, kernel, iterations = 3)

    # display image
    cv2.imshow("Closing - Fill in blobs", closing)
    cv2.waitKey(3)


if __name__=='__main__':
    main()
