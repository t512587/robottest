import numpy as np
import cv2
import json
import pyrealsense2 as rs
from pymycobot.elephantrobot import ElephantRobot
from scipy.spatial.transform import Rotation as R
import time
import threading
from collections import deque

class RealTimeCoordinateTransform:
    def __init__(self, camera_to_gripper_matrix, robot_ip="192.168.1.159", robot_port=5001):
        """
        初始化實時座標轉換系統
        
        Args:
            camera_to_gripper_matrix: 4x4 相機到夾爪的變換矩陣
            robot_ip: 機械手臂IP地址
            robot_port: 機械手臂端口
        """
        # 相機到夾爪的變換矩陣
        self.T_camera_gripper = np.array(camera_to_gripper_matrix)
        
        # 機械手臂連接
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.robot = None
        self.robot_connected = False
        
        # 相機參數
        self.camera_matrix = np.array([[616.798, 0, 321.753],
                                      [0, 616.904, 247.541],
                                      [0, 0, 1]])
        self.dist_coeffs = np.zeros((5, 1))
        
        # ArUco 設定
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.marker_length = 0.04  # 4cm標記大小
        
        # 相機管線
        self.pipeline = None
        self.config = None
        
        # 轉換結果緩存
        self.current_transforms = {}
        self.transform_history = deque(maxlen=10)  # 保存最近10次轉換結果
        
        # 控制變數
        self.running = False
        self.show_debug_info = True
        self.auto_move_enabled = False
        self.target_marker_id = None
        
        print("實時座標轉換系統初始化完成")
    
    def connect_robot(self):
        """連接機械手臂"""
        try:
            self.robot = ElephantRobot(self.robot_ip, self.robot_port)
            self.robot.start_client()
            self.robot_connected = True
            print(f"機械手臂連接成功: {self.robot_ip}:{self.robot_port}")
            return True
        except Exception as e:
            print(f"機械手臂連接失敗: {e}")
            self.robot_connected = False
            return False
    
    def init_camera(self):
        """初始化相機"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(self.config)
            print("RealSense相機初始化成功")
            return True
        except Exception as e:
            print(f"相機初始化失敗: {e}")
            return False
    
    def invert_transform(self, T):
        """反轉4x4齊次變換矩陣"""
        R = T[:3, :3]
        t = T[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T_inv
    
    def transform_point(self, T, P):
        """用4x4變換矩陣轉換3D點"""
        P_h = np.ones(4)
        P_h[:3] = P
        P_transformed = T @ P_h
        return P_transformed[:3]
    
    def camera_to_base_transform(self, camera_point, gripper_pose):
        """
        將相機座標轉換為基座座標
        
        Args:
            camera_point: 物體在相機座標系的位置 [x, y, z] (mm)
            gripper_pose: 當前夾爪姿態 [x, y, z, rx, ry, rz]
        
        Returns:
            物體在基座座標系的位置 [x, y, z] (mm)
        """
        # 構建基座到夾爪的變換矩陣
        x, y, z, rx, ry, rz = gripper_pose
        
        # 位置向量
        translation = np.array([x, y, z])
        
        # 旋轉矩陣 (從度轉換為弧度)
        rotation = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        rotation_matrix = rotation.as_matrix()
        
        # 組成4x4變換矩陣
        T_base_gripper = np.eye(4)
        T_base_gripper[:3, :3] = rotation_matrix
        T_base_gripper[:3, 3] = translation
        
        # 計算夾爪到相機的變換
        T_gripper_camera = self.invert_transform(self.T_camera_gripper)
        
        # 計算物體在基座座標系的位置
        # P_base = T_base_gripper * T_gripper_camera * P_camera
        P_base = self.transform_point(T_base_gripper @ T_gripper_camera, camera_point)
        
        return P_base
    
    def detect_aruco_markers(self, color_image):
        """檢測ArUco標記並計算其3D位置"""
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # 檢測標記
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters)
        
        detected_markers = {}
        
        if ids is not None:
            # 估計姿態
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            for i in range(len(ids)):
                marker_id = ids[i][0]
                camera_pos = tvecs[i][0] * 1000  # 轉換為mm
                
                detected_markers[marker_id] = {
                    'camera_position': camera_pos,
                    'corners': corners[i],
                    'rvec': rvecs[i],
                    'tvec': tvecs[i]
                }
        
        return detected_markers, corners, ids
    
    def draw_detection_results(self, color_image, detected_markers, corners, ids):
        """在影像上繪製檢測結果"""
        if ids is not None:
            # 繪製檢測到的標記
            cv2.aruco.drawDetectedMarkers(color_image, corners)
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in detected_markers:
                    marker_data = detected_markers[marker_id]
                    
                    # 繪製坐標軸
                    cv2.drawFrameAxes(color_image, self.camera_matrix, 
                                    self.dist_coeffs, marker_data['rvec'], 
                                    marker_data['tvec'], 0.03)
                    
                    # 顯示資訊
                    corner = corners[i][0][0]
                    x, y = int(corner[0]), int(corner[1])
                    
                    # 標記ID
                    cv2.putText(color_image, f"ID: {marker_id}", 
                              (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 相機座標
                    cam_pos = marker_data['camera_position']
                    cv2.putText(color_image, f"Cam: ({cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f})", 
                              (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    
                    # 基座座標（如果有機械手臂連接）
                    if self.robot_connected and marker_id in self.current_transforms:
                        base_pos = self.current_transforms[marker_id]['base_position']
                        cv2.putText(color_image, f"Base: ({base_pos[0]:.0f}, {base_pos[1]:.0f}, {base_pos[2]:.0f})", 
                                  (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        
                        # 標示目標標記
                        if marker_id == self.target_marker_id:
                            cv2.rectangle(color_image, (x-10, y-80), (x+200, y+10), (0, 255, 255), 2)
                            cv2.putText(color_image, "TARGET", (x, y-65), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def draw_status_info(self, color_image):
        """繪製系統狀態資訊"""
        height, width = color_image.shape[:2]
        
        # 背景框
        cv2.rectangle(color_image, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(color_image, (10, 10), (400, 150), (255, 255, 255), 2)
        
        y_offset = 30
        
        # 系統狀態
        status_color = (0, 255, 0) if self.robot_connected else (0, 0, 255)
        robot_status = "連接" if self.robot_connected else "未連接"
        cv2.putText(color_image, f"Robot Arm: {robot_status}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y_offset += 25
        
        # 當前機械手臂位置
        if self.robot_connected:
            try:
                current_coords = self.robot.get_coords()
                cv2.putText(color_image, f"Current position: ({current_coords[0]:.0f}, {current_coords[1]:.0f}, {current_coords[2]:.0f})", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                y_offset += 20
            except:
                pass
        
        # 控制說明
        cv2.putText(color_image, "Key Control:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(color_image, "C: Get coordinates  M: Move to target  Q: Quit", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 15
        cv2.putText(color_image, "Number keys: Select target marker ID", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def move_to_target(self, target_position, speed=30):
        """移動機械手臂到目標位置"""
        if not self.robot_connected:
            print("機械手臂未連接")
            return False
        
        try:
            # 獲取當前位置和姿態
            current_coords = self.robot.get_coords()
            
            # 保持當前姿態，只改變位置
            target_coords = [
                target_position[0],
                target_position[1], 
                target_position[2],
                current_coords[3],  # 保持原有的rx
                current_coords[4],  # 保持原有的ry
                current_coords[5]   # 保持原有的rz
            ]
            
            print(f"移動到目標位置: {target_coords[:3]}")
            self.robot.send_coords(target_coords, speed)
            return True
            
        except Exception as e:
            print(f"移動失敗: {e}")
            return False
    
    def run(self):
        """運行實時轉換系統"""
        # 初始化相機
        if not self.init_camera():
            return
        
        # 連接機械手臂
        self.connect_robot()
        
        self.running = True
        print("\n=== 實時座標轉換系統啟動 ===")
        print("操作說明:")
        print("- 按 'C' 鍵: 獲取當前檢測到的座標")
        print("- 按 'M' 鍵: 移動機械手臂到目標位置")
        print("- 按數字鍵 (0-9): 選擇目標標記ID")
        print("- 按 'Q' 鍵: 退出系統")
        
        try:
            while self.running:
                # 獲取影像
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                # 檢測ArUco標記
                detected_markers, corners, ids = self.detect_aruco_markers(color_image)
                
                # 如果有機械手臂連接，計算基座座標
                if self.robot_connected and detected_markers:
                    try:
                        current_gripper_pose = self.robot.get_coords()
                        
                        for marker_id, marker_data in detected_markers.items():
                            camera_pos = marker_data['camera_position']
                            base_pos = self.camera_to_base_transform(camera_pos, current_gripper_pose)
                            
                            self.current_transforms[marker_id] = {
                                'camera_position': camera_pos,
                                'base_position': base_pos,
                                'timestamp': time.time()
                            }
                    except Exception as e:
                        if self.show_debug_info:
                            print(f"座標轉換錯誤: {e}")
                
                # 繪製檢測結果
                self.draw_detection_results(color_image, detected_markers, corners, ids)
                
                # 繪製狀態資訊
                if self.show_debug_info:
                    self.draw_status_info(color_image)
                
                # 顯示影像
                cv2.imshow('Real-time Coordinate Transformation System', color_image)
                
                # 處理按鍵
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    self.running = False
                    break
                    
                elif key == ord('c') or key == ord('C'):
                    self.print_current_transforms()
                    
                elif key == ord('m') or key == ord('M'):
                    if self.target_marker_id is not None and self.target_marker_id in self.current_transforms:
                        target_pos = self.current_transforms[self.target_marker_id]['base_position']
                        self.move_to_target(target_pos)
                    else:
                        print("請先選擇目標標記ID（按數字鍵0-9）")
                        
                elif key >= ord('0') and key <= ord('9'):
                    self.target_marker_id = key - ord('0')
                    print(f"目標標記ID設定為: {self.target_marker_id}")
                    
                elif key == ord('d') or key == ord('D'):
                    self.show_debug_info = not self.show_debug_info
                    print(f"調試資訊顯示: {'開啟' if self.show_debug_info else '關閉'}")
        
        except KeyboardInterrupt:
            print("\n收到中斷信號，正在關閉系統...")
        
        finally:
            self.cleanup()
    
    def print_current_transforms(self):
        """印出當前的座標轉換結果"""
        if not self.current_transforms:
            print("目前沒有檢測到ArUco標記")
            return
        
        print("\n=== 當前座標轉換結果 ===")
        for marker_id, transform_data in self.current_transforms.items():
            camera_pos = transform_data['camera_position']
            base_pos = transform_data['base_position']
            
            print(f"標記 ID {marker_id}:")
            print(f"  相機座標: ({camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f}) mm")
            print(f"  基座座標: ({base_pos[0]:.1f}, {base_pos[1]:.1f}, {base_pos[2]:.1f}) mm")
            print(f"  時間戳: {time.strftime('%H:%M:%S', time.localtime(transform_data['timestamp']))}")
            print("-" * 50)
    
    def cleanup(self):
        """清理資源"""
        print("正在清理資源...")
        
        if self.pipeline:
            self.pipeline.stop()
        
        cv2.destroyAllWindows()
        
        if self.robot_connected and self.robot:
            try:
                self.robot.close()
            except:
                pass
        
        print("系統已關閉")


def main():
    """主程式"""
    # 手眼標定結果的變換矩陣（相機到夾爪）
    T_camera_gripper = np.array([
    [ 2.57331544e-01,  6.30717308e-03, -9.66302590e-01,  7.86118890e+01],
    [-8.37292439e-01,  5.00669529e-01, -2.19707521e-01,  4.61975737e+01],
    [ 4.82412529e-01,  8.65615528e-01,  1.34119011e-01, -4.39087375e+01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    ])
    
    print("=== 實時相機到機械手臂座標轉換系統 ===")
    print("載入手眼標定矩陣...")
    print("變換矩陣 (相機到夾爪):")
    print(T_camera_gripper)
    
    # 創建實時轉換系統
    transform_system = RealTimeCoordinateTransform(
        camera_to_gripper_matrix=T_camera_gripper,
        robot_ip="192.168.1.159",  # 請根據實際情況修改
        robot_port=5001
    )
    
    # 運行系統
    transform_system.run()


if __name__ == "__main__":
    main()