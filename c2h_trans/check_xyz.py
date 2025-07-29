import cv2
import numpy as np
import pyrealsense2 as rs

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
aruco_params = cv2.aruco.DetectorParameters()

clicked_points = []
depth_frame_global = None
camera_intrinsics = None

# 新增一個列表保存每一幀偵測到的ArUco座標(避免每幀都刷新，可以只保留最後一幀)
aruco_detected_points = []

def mouse_callback(event, x, y, flags, param):
    global depth_frame_global, camera_intrinsics
    if event == cv2.EVENT_LBUTTONDOWN:
        if depth_frame_global is None:
            print("[錯誤] 尚未獲取深度資訊")
            return

        depth = depth_frame_global.get_distance(x, y)
        if depth <= 0:
            print(f"[錯誤] 無效深度：({x}, {y})")
            return

        point_3d = rs.rs2_deproject_pixel_to_point(camera_intrinsics, [x, y], depth)
        point_3d_mm = [coord * 1000 for coord in point_3d]

        clicked_points.append({
            "pixel": (x, y),
            "camera_coord": point_3d_mm
        })

        print(f"[人工點擊] 像素: ({x}, {y}) → 相機座標: ({point_3d_mm[0]:.1f}, {point_3d_mm[1]:.1f}, {point_3d_mm[2]:.1f}) mm")

def detect_aruco_centers(image, depth_frame):
    global aruco_detected_points
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    centers_3d = []

    if ids is not None:
        for i, corner in enumerate(corners):
            pts = corner[0]
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))

            depth = depth_frame.get_distance(center_x, center_y)
            if depth > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(camera_intrinsics, [center_x, center_y], depth)
                point_3d_mm = [coord * 1000 for coord in point_3d]
                centers_3d.append((ids[i][0], point_3d_mm))

                cv2.polylines(image, [pts.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(image, f"ID:{ids[i][0]}", (center_x + 5, center_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(image, f"({int(point_3d_mm[0])},{int(point_3d_mm[1])},{int(point_3d_mm[2])})",
                            (center_x + 5, center_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
                print(center_x, center_y, depth)
    aruco_detected_points = centers_3d  # 每次更新最新偵測到的ArUco座標
    return image, centers_3d

def main():
    global depth_frame_global, camera_intrinsics

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    camera_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

    cv2.namedWindow("RealSense ArUco")
    cv2.setMouseCallback("RealSense ArUco", mouse_callback)

    print("左鍵點擊取得人工點擊相機座標，按 's' 儲存點，按 'q' 結束")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            depth_frame_global = depth_frame
            color_image = np.asanyarray(color_frame.get_data())

            display_img, aruco_centers = detect_aruco_centers(color_image.copy(), depth_frame)

            for pt in clicked_points:
                px = pt["pixel"]
                xyz = pt["camera_coord"]
                cv2.circle(display_img, px, 5, (255, 0, 255), -1)
                cv2.putText(display_img, f"M:({int(xyz[0])},{int(xyz[1])},{int(xyz[2])})",
                            (px[0]+5, px[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)

            cv2.imshow("RealSense ArUco", display_img)
            key = cv2.waitKey(1)

            if key == ord('q'):
                # 離開前印出所有自動偵測及人工點擊的座標
                print("\n=== 最終輸出 ===")
                print("自動偵測 ArUco 相機座標：")
                if len(aruco_detected_points) == 0:
                    print("  無偵測到 ArUco")
                else:
                    for id_, coord in aruco_detected_points:
                        print(f"  ID:{id_} → 相機座標: ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}) mm")

                print("\n人工點擊點座標：")
                if len(clicked_points) == 0:
                    print("  無人工點擊")
                else:
                    for i, pt in enumerate(clicked_points):
                        x, y = pt["pixel"]
                        cx, cy, cz = pt["camera_coord"]
                        print(f"  點{i+1}: 像素=({x}, {y}) → 相機座標=({cx:.1f}, {cy:.1f}, {cz:.1f}) mm")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
