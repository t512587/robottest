import pyrealsense2 as rs
import numpy as np
import cv2
import os

output_dir = 'snapshots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 自動接續 snapshot 編號 ---
existing_snapshots = [d for d in os.listdir(output_dir) if d.startswith("snapshot_")]
existing_indices = [int(name.split('_')[-1]) for name in existing_snapshots if name.split('_')[-1].isdigit()]
snapshot_counter = max(existing_indices) + 1 if existing_indices else 1

# --- 初始化 RealSense ---
pipeline = rs.pipeline()
config = rs.config()

# 啟用深度與彩色流
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 建立對齊工具
align_to = rs.stream.color
align = rs.align(align_to)

# 啟動攝影機
pipeline.start(config)
print("攝影機啟動中，請稍候...")

# 丟棄最初的幾幀，以等待曝光穩定
for i in range(30):
    pipeline.wait_for_frames()

print("準備完成！看著預覽視窗，按下 's' 鍵拍照，按下 'q' 鍵退出。")

try:
    while True:
        # 擷取影像並對齊
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # 轉為 numpy 陣列
        # Shape	(480, 640)
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # Shape	(480, 640, 3)
        color_image = np.asanyarray(color_frame.get_data())

        # # 偽彩色深度圖
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        cv2.imshow('RealSense 預覽 (按 s 拍照, 按 q 退出)', color_image)
        key = cv2.waitKey(1)

        if key == ord('s'):
            snapshot_folder = f"snapshot_{snapshot_counter:04d}"
            current_dir = os.path.join(output_dir, snapshot_folder)
            os.makedirs(current_dir)

            color_path = os.path.join(current_dir, 'color.png')
            depth_raw_path = os.path.join(current_dir, 'depth_raw.png')
            depth_colormap_path = os.path.join(current_dir, 'depth_colormap.png')
            depth_numpy_path = os.path.join(current_dir, 'depth_data.npy')

            cv2.imwrite(color_path, color_image)
            cv2.imwrite(depth_colormap_path, depth_colormap)
            cv2.imwrite(depth_raw_path, depth_image)
            np.save(depth_numpy_path, depth_image)

            print(f"照片已儲存至 '{current_dir}' !")
            snapshot_counter += 1

        elif key == ord('q'):
            print("正在關閉程式...")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
