import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import time
import os
from pipeline import ImageRetrievalPipeline
from config import Config
from pymycobot.elephantrobot import ElephantRobot, JogMode
from scipy.spatial.transform import Rotation as R

# 🔁 4x4 變換矩陣轉換與反轉
def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def transform_point(T, P):
    P_h = np.ones(4)
    P_h[:3] = P
    P_transformed = T @ P_h
    return P_transformed[:3]

# 📷 相機內參與座標轉換
def pixel_to_camera_coords(cx, cy, depth_mm, fx, fy, cx_cam, cy_cam):
    X = (cx - cx_cam) * depth_mm / fx
    Y = (cy - cy_cam) * depth_mm / fy
    Z = depth_mm
    return np.array([X, Y, Z])

# MODIFIED: 新增函數，獲取完整相機內參及畸變係數
def get_camera_intrinsics_and_distortion(frames):
    color_frame = frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().get_intrinsics()

    # 從 intrinsics 物件中提取焦距和光學中心
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy

    # 建立相機矩陣 (Camera Matrix)
    camera_matrix = np.array([
        [fx, 0, ppx],
        [0, fy, ppy],
        [0, 0, 1]
    ], dtype=np.float32)

    # 獲取畸變係數 (Distortion Coefficients)
    dist_coeffs = np.array(intr.coeffs, dtype=np.float32)

    # 返回相機矩陣、畸變係數、以及單獨的 fx, fy, ppx, ppy 以便舊程式碼兼容
    return camera_matrix, dist_coeffs, fx, fy, ppx, ppy

def get_median_depth(depth_img, cx, cy, window=5):
    h, w = depth_img.shape
    x1, x2 = max(cx - window // 2, 0), min(cx + window // 2 + 1, w)
    y1, y2 = max(cy - window // 2, 0), min(cy + window // 2 + 1, h)
    patch = depth_img[y1:y2, x1:x2]
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None
    return np.median(valid)

# 📦 初始化檢索模型與手臂
retrieval_pipeline = ImageRetrievalPipeline(Config())
retrieval_pipeline.build_database()

elephant_client = ElephantRobot("192.168.1.159", 5001)
elephant_client.start_client()

# 📷 初始化 RealSense pipeline
config1 = rs.config()
config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline1 = rs.pipeline()
profile = pipeline1.start(config1)
align1 = rs.align(rs.stream.color)

frames = pipeline1.wait_for_frames()
aligned_frames = align1.process(frames)

# MODIFIED: 呼叫新的函數來獲取所有內參
camera_matrix, dist_coeffs, fx, fy, cx_cam, cy_cam = get_camera_intrinsics_and_distortion(aligned_frames)

# MODIFIED: 印出相機內參和畸變係數
print("--- RealSense 相機內參 ---")
print("相機矩陣 (Camera Matrix, K):")
print(camera_matrix)
print("\n畸變係數 (Distortion Coefficients):")
print(dist_coeffs)
print("---------------------------\n")


# 🔁 相機到夾爪的固定變換矩陣（Tsai-Lenz 手眼標定結果）
T_camera_gripper = np.array([
 [ 2.90836209e-01,  7.64953618e-02, -9.53709998e-01,  4.46063561e+02],
 [ 6.43395326e-03,  9.96619855e-01,  8.18991341e-02,  6.18706117e+01],
 [ 9.56751224e-01, -2.99553592e-02,  2.89360973e-01,  3.59376086e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
])
T_gripper_camera = invert_transform(T_camera_gripper)

# 📂 確保儲存資料夾存在
os.makedirs("output_images", exist_ok=True)

print("📷 按下 's' 進行商品偵測與位置轉換")
print("Q: 離開程式")

try:
    while True:
        frames = pipeline1.wait_for_frames()
        aligned_frames = align1.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image_raw = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # MODIFIED: 使用實際的相機矩陣和畸變係數對彩色圖像進行去畸變
        # 這一步很重要，確保後續的物體偵測和座標計算是在無畸變的圖像上進行的
        color_image_undistorted = cv2.undistort(color_image_raw, camera_matrix, dist_coeffs)

        cv2.imshow("RealSense", color_image_undistorted) # 顯示去畸變後的圖像
        key = cv2.waitKey(1)

        if key == ord('s'):
            print("🔍 開始商品偵測")
            # MODIFIED: 對物體偵測使用去畸變後的圖像
            rgb_img = cv2.cvtColor(color_image_undistorted, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            filename, objects = retrieval_pipeline.yolo_parser.detect_objects(pil_img, conf_threshold=0.7)

            if not objects:
                print("⚠️ 沒有偵測到物件")
                continue

            predictions = []
            for obj in objects:
                cropped = pil_img.crop((obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']))
                results = retrieval_pipeline.retriever.retrieve_similar_images(
                    cropped,
                    retrieval_pipeline.feature_db,
                    retrieval_pipeline.name_db,
                    retrieval_pipeline.label_db,
                    topk=retrieval_pipeline.config.TOP_K
                )
                predictions.append(results)

            # MODIFIED: 在去畸變後的圖像上進行標註和儲存
            annotated_image = color_image_undistorted.copy()

            for i, obj in enumerate(objects):
                xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2

                top1_result = predictions[i][0]
                label_id = top1_result['label']
                label_name = retrieval_pipeline.config.ID2LABEL.get(label_id, "unknown")

                # 畫框 + 中心點 + label
                cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.circle(annotated_image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(annotated_image, f"{label_name}", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                depth_median = get_median_depth(depth_image, cx, cy)
                if depth_median is None:
                    print(f"[WARN] 第{i+1}個目標 [{label_name}] 深度無效，跳過")
                    continue

                # 像素到相機座標的轉換使用從 SDK 獲取的 fx, fy, cx_cam, cy_cam
                cam_coords = pixel_to_camera_coords(cx, cy, depth_median, fx, fy, cx_cam, cy_cam)
                print(f"📍 第{i+1}個目標 [{label_name}] 相機座標：", cam_coords)

                current_coords = elephant_client.get_coords()
                xyz = current_coords[:3]
                rpy = current_coords[3:]  # 單位是度

                # 構建 Base ➜ Gripper 的完整 4x4 變換矩陣
                T_base_gripper = np.eye(4)
                T_base_gripper[:3, 3] = np.array(xyz)

                # 加入 RPY 角度（轉為旋轉矩陣）
                rotation = R.from_euler('xyz', rpy, degrees=True).as_matrix()

                T_base_gripper[:3, :3] = rotation

                # 將相機座標轉換為基座座標
                P_base = transform_point(T_base_gripper @ T_gripper_camera, cam_coords)
                print(f"🎯 第{i+1}個目標 [{label_name}] 基座座標：", P_base)


            # 儲存圖像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"output_images/detected_{timestamp}.jpg"
            cv2.imwrite(save_path, annotated_image)
            print(f"✅ 已儲存視覺化圖像：{save_path}")

        elif key == ord('q'):
            print("👋 離開程式")
            break

except KeyboardInterrupt:
    print("🛑 程式中斷")

finally:
    pipeline1.stop()
    cv2.destroyAllWindows()
    # 假設你用 T_camera_gripper 轉換 gripper 的前方向 (1, 0, 0)
    # 看轉出來是不是 camera 的 Z 軸 (0, 0, 1)
    forward_in_gripper = np.array([1, 0, 0])
    forward_in_camera = T_camera_gripper[:3, :3] @ forward_in_gripper
    print("Gripper 向前方向轉到相機座標系變成：", forward_in_camera)
    gripper_y = np.array([0,1,0])
    gripper_z = np.array([0,0,1])

    cam_y = T_camera_gripper[:3, :3] @ gripper_y
    cam_z = T_camera_gripper[:3, :3] @ gripper_z

    print("Gripper Y 軸轉到相機座標系：", cam_y)
    print("Gripper Z 軸轉到相機座標系：", cam_z) 