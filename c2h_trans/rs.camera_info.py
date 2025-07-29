import pyrealsense2 as rs

"""ctx = rs.context()
devices = ctx.query_devices()
print("找到的 RealSense 裝置:")
for i, dev in enumerate(devices):
    print(f"裝置 {i}: {dev.get_info(rs.camera_info.serial_number)}")"""
from pymycobot.elephantrobot import ElephantRobot
import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import json
import os

# 建立與機械臂連線
elephant_client = ElephantRobot("192.168.1.159", 5001)
print(dir(elephant_client))