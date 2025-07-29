import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import json
import os
from pymycobot.elephantrobot import ElephantRobot

# === 初始化連線 ===
elephant_client = ElephantRobot("192.168.1.159", 5001)
elephant_client.start_client()
print("ElephantRobot目前座標：", elephant_client.get_coords())

# === 初始化 RealSense 相機 ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# === 取得內參 ===
profile = pipeline.get_active_profile()
video_stream_profile = profile.get_stream(rs.stream.color)
intr = video_stream_profile.as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
])
dist_coeffs = np.zeros((5, 1))

# === ArUco 設定 ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
marker_length = 0.04  # 單位: 公尺

# === 資料儲存 ===
output_data = []
save_dir = "handeye_records"
os.makedirs(save_dir, exist_ok=True)

print("\n=== 手眼標定資料記錄系統 ===")
print("s - 記錄 ArUco + 機械手初始姿態")
print("m - 記錄該點移動後的姿態")
print("v - 查看已記錄資料")
print("r - 重置資料")
print("q - 離開並儲存")
print("=" * 40)

try:
    current_index = 0

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(color_image, corners)
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)
                corner = corners[i][0]
                cv2.putText(color_image, f"ID: {ids[i][0]}",
                            (int(corner[0][0]), int(corner[0][1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Hand-Eye Calibration Collector", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s') and ids is not None:
            # 只記錄第一個 ArUco
            rvec = rvecs[0][0].tolist()
            tvec = (tvecs[0][0] * 1000).tolist()  # 將 m 轉換為 mm
            marker_id = int(ids[0][0])
            robot_pose = elephant_client.get_coords()

            entry = {
                "marker_id": marker_id,
                "aruco_tvec": tvec,
                "aruco_rvec": rvec,
                "robot_pose_at_detect": robot_pose,
                "robot_pose_after_move": None
            }

            output_data.append(entry)
            current_index = len(output_data) - 1

            print(f"\n✅ 已記錄第 {current_index + 1} 筆 (ID: {marker_id})")
            print(f"ArUco tvec (m): {tvec}")
            print(f"ArUco rvec: {rvec}")
            print(f"手臂姿態: {robot_pose}")

        elif key == ord('m') and output_data and output_data[current_index]["robot_pose_after_move"] is None:
            moved_pose = elephant_client.get_coords()
            output_data[current_index]["robot_pose_after_move"] = moved_pose
            print(f"🔁 已補上移動後手臂姿態：{moved_pose}")

        elif key == ord('v'):
            print(f"\n📋 已記錄 {len(output_data)} 筆資料：")
            for i, d in enumerate(output_data):
                moved = "✅" if d["robot_pose_after_move"] else "⏳"
                print(f"  第 {i+1} 筆 - ID: {d['marker_id']} {moved}")

        elif key == ord('r'):
            output_data = []
            print("🔄 已清空所有記錄")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    if output_data:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"handeye_record_{timestamp}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 已儲存 {len(output_data)} 筆資料至 {filename}")
    print("📌 程式結束")
