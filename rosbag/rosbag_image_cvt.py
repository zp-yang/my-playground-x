from itertools import count
import numpy as np
import rosbag
from cv_bridge import CvBridge
import cv2
import glob
import os

out_dir = "/home/zp-yang/rosbag/datasets/"
bag_files = glob.glob("/home/zp-yang/rosbag/*.bag")
print(bag_files)


bridge = CvBridge()

for i in range(4):
    bag = rosbag.Bag(bag_files[i])

    ct = 0

    cur_out_dir = out_dir + "dataset_{}".format(i)
    print("current output directory: {}".format(cur_out_dir))

    for topic, msg, t in bag.read_messages(topics=["/camera_0/image_raw"]):
        if ct % 5 == 0:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv2.imwrite("{}/frame-{}.png".format(cur_out_dir, ct), cv_img)
        ct += 1

    print(ct)