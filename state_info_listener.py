#!/usr/bin/env python

import sys
import copy
import rospy
import numpy as np
from tf import transformations as tfs
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
from math import pi
from std_msgs.msg import String, Header

from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, WrenchStamped
from ur_control_wrapper.srv import GetWrench, GetWrenchResponse
from ur_control_wrapper.srv import GetCamImage, GetCamImageResponse
from ur_control_wrapper.srv import GetAlignedDepthImg, GetAlignedDepthImgResponse
from cv_bridge import CvBridge, CvBridgeError


class StateListener:

    def __init__(self):

        # cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        # cv_depth = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
        self.bridge = CvBridge()

        self.wrench = None
        self.wrench_sub = rospy.Subscriber('/wrench', WrenchStamped, self.wrench_callback, queue_size=1)

        self.img = (np.zeros((480,640)))
        self.img_raw = None
        self.cam_image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        self.img_depth = None
        self.cam_image_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.image_depth_callback)

        rospy.Service('ur_control_wrapper/get_wrench', GetWrench, self.get_wrench)
        rospy.Service('ur_control_wrapper/get_cam_image', GetCamImage, self.get_cam_image)
        rospy.Service('ur_control_wrapper/get_aligned_depth_img', GetAlignedDepthImg, self.get_aligned_depth_img)

    
    def wrench_callback(self, data):
        self.wrench = data


    def image_callback(self,img):
        self.img_raw = img

    
    def image_depth_callback(self, img):
        self.img_depth = img


    def get_wrench(self, data):
        return GetWrenchResponse(self.wrench)


    def get_cam_image(self, data):
        return GetCamImageResponse(self.img_raw)

    
    def get_aligned_depth_img(self, data):
        return GetAlignedDepthImgResponse(self.img_depth)


if __name__ == '__main__':
    try:
        rospy.init_node('ur_control_wrapper_statespace_listener_mode', anonymous=True)
        state_listener = StateListener()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Statespace listener not started!!")
        pass
