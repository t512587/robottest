import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco

# RealSense pipeline設定
pipeline = rs.pipeline()
config = rs.config()

# 啟用彩色影像串流（640x480, 30fps）
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 開始串流
pipeline.start(config)

# ArUco參數設定
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
parameters = aruco.DetectorParameters_create()

# 相機內參（建議自己校正，以下示例請換成你自己的）
camera_matrix = np.array([
    [925.1968994140625, 0, 642.629638671875],
    [0, 925.3560791015625, 371.3112487792969],
    [0, 0, 1]
])
dist_coeffs = np.zeros((5, 1))  # 假設無畸變

marker_length = 0.05  # 5cm標記邊長（單位公尺）

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 轉成numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # 轉灰階
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 偵測 ArUco 標記
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # 畫出標記邊框
            aruco.drawDetectedMarkers(color_image, corners, ids)

            # 計算標記姿態
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs)

            for rvec, tvec in zip(rvecs, tvecs):
                aruco.drawAxis(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                print("rvec:", rvec.flatten())
                print("tvec:", tvec.flatten())
                print("---")

        cv2.imshow('RealSense ArUco Detection', color_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
