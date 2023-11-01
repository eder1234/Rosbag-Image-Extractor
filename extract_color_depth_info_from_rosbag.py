#!/usr/bin/python2.7

import os
import sys
import rospy
import rosbag
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

def extract_images_from_bag(bag_file, output_folder):
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    color_count = 0
    depth_count = 0

    # Make directories for color and depth images
    color_folder = os.path.join(output_folder, "color")
    depth_folder = os.path.join(output_folder, "depth")
    if not os.path.exists(color_folder):
        os.makedirs(color_folder)
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)

    for topic, msg, t in bag.read_messages():
        try:
            if topic == "/device_0/sensor_1/Color_0/image/data":
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                image_filename = os.path.join(color_folder, "frame_{:04d}.png".format(color_count))
                cv2.imwrite(image_filename, cv_image)
                print("Saved Color: {}".format(image_filename))
                color_count += 1
            elif topic == "/device_0/sensor_0/Depth_0/image/data":
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")  # Depth images are usually 16-bit monochrome
                image_filename = os.path.join(depth_folder, "frame_{:04d}.png".format(depth_count))
                cv2.imwrite(image_filename, cv_image)
                print("Saved Depth: {}".format(image_filename))
                depth_count += 1
            elif topic == "/device_0/sensor_0/Depth_0/info/camera_info":
                save_camera_info_to_txt(msg, os.path.join(output_folder, "Depth_Camera_Info.txt"))
            elif topic == "/device_0/sensor_1/Color_0/info/camera_info":
                save_camera_info_to_txt(msg, os.path.join(output_folder, "Color_Camera_Info.txt"))
        except Exception as e:
            print("Error extracting data: {}".format(e))

    bag.close()

def save_camera_info_to_txt(camera_info_msg, output_file_path):
    with open(output_file_path, "w") as file:
        file.write("Image width: {}\n".format(camera_info_msg.width))
        file.write("Image height: {}\n".format(camera_info_msg.height))
        file.write("Camera model: {}\n".format(camera_info_msg.distortion_model))
        file.write("Camera matrix (K):\n")
        for i in range(3):
            file.write(" ".join(str(x) for x in camera_info_msg.K[i*3:(i+1)*3]) + "\n")
        file.write("Distortion coefficients (D):\n")
        file.write(" ".join(str(x) for x in camera_info_msg.D) + "\n")
        file.write("Rectification matrix (R):\n")
        for i in range(3):
            file.write(" ".join(str(x) for x in camera_info_msg.R[i*3:(i+1)*3]) + "\n")
        file.write("Projection matrix (P):\n")
        for i in range(3):
            file.write(" ".join(str(x) for x in camera_info_msg.P[i*4:(i+1)*4]) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_images_from_rosbag.py <path_to_bag> <output_folder>")
        sys.exit(1)

    bag_file = sys.argv[1]
    output_folder = sys.argv[2]
    
    extract_images_from_bag(bag_file, output_folder)
