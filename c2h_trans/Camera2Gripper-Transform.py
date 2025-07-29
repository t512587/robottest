import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class HandEyeCalibrationProcessor:
    def __init__(self):
        self.calibration_data = []
    
    def load_json_data(self, json_data):
        """
        載入JSON標定數據（可以是單組或多組數據）
        """
        if isinstance(json_data, str):
            # 如果是JSON字符串，解析它
            data = json.loads(json_data)
        else:
            # 如果已經是字典格式
            data = json_data
        
        return data
    
    def load_json_file_with_multiple_records(self, file_path):
        """
        載入包含多組標定記錄的JSON文件
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 檢查數據格式
            if isinstance(data, list):
                # 如果是列表格式，每個元素是一組標定數據
                print(f"檢測到列表格式，包含 {len(data)} 組標定數據")
                for i, record in enumerate(data):
                    print(f"處理第 {i+1} 組數據...")
                    self.add_calibration_point(record)
                    
            elif isinstance(data, dict):
                # 檢查是否包含多組數據的字典格式
                if 'records' in data or 'calibration_data' in data:
                    # 如果有 records 或 calibration_data 鍵
                    records_key = 'records' if 'records' in data else 'calibration_data'
                    records = data[records_key]
                    print(f"檢測到字典格式，包含 {len(records)} 組標定數據")
                    for i, record in enumerate(records):
                        print(f"處理第 {i+1} 組數據...")
                        self.add_calibration_point(record)
                elif all(key in data for key in ['marker_id', 'aruco_tvec', 'aruco_rvec', 'robot_pose_at_detect']):
                    # 如果是單組數據格式
                    print("檢測到單組標定數據")
                    self.add_calibration_point(data)
                else:
                    # 可能是以數字為鍵的字典格式
                    numeric_keys = [key for key in data.keys() if str(key).isdigit()]
                    if numeric_keys:
                        print(f"檢測到數字鍵格式，包含 {len(numeric_keys)} 組標定數據")
                        for key in sorted(numeric_keys, key=int):
                            print(f"處理第 {key} 組數據...")
                            self.add_calibration_point(data[key])
                    else:
                        raise ValueError("無法識別的JSON數據格式")
            else:
                raise ValueError("JSON數據必須是字典或列表格式")
                
            print(f"成功載入 {len(self.calibration_data)} 組標定數據")
            return len(self.calibration_data)
            
        except FileNotFoundError:
            print(f"錯誤: 找不到文件 {file_path}")
            return 0
        except json.JSONDecodeError as e:
            print(f"錯誤: JSON格式解析失敗 - {e}")
            return 0
        except Exception as e:
            print(f"錯誤: {e}")
            return 0
    
    def rodrigues_to_matrix(self, rvec):
        """
        將Rodrigues向量轉換為旋轉矩陣
        """
        rotation_matrix, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
        return rotation_matrix
    
    def pose_to_transformation_matrix(self, pose):
        """
        將6DOF姿態轉換為4x4變換矩陣
        pose: [x, y, z, rx, ry, rz] (位置單位：mm，角度單位：度)
        """
        x, y, z, rx, ry, rz = pose
        
        # 位置向量 (轉換為米)
        #translation = np.array([x/1000, y/1000, z/1000])
        translation = np.array([x, y, z])
        
        # 旋轉矩陣 (從度轉換為弧度)
        rotation = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        rotation_matrix = rotation.as_matrix()
        
        # 組成4x4變換矩陣
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation
        
        return T
    
    def aruco_to_transformation_matrix(self, tvec, rvec):
        """
        將ArUco檢測結果轉換為4x4變換矩陣
        """
        # 旋轉矩陣
        rotation_matrix = self.rodrigues_to_matrix(rvec)
        
        # 組成4x4變換矩陣
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = np.array(tvec)
        
        return T
    
    def process_calibration_data(self, json_data):
        """
        處理單筆標定數據
        """
        data = self.load_json_data(json_data)
        
        # 提取數據
        marker_id = data['marker_id']
        aruco_tvec = data['aruco_tvec']
        aruco_rvec = data['aruco_rvec']
        robot_pose = data['robot_pose_at_detect']
        
        print(f"處理標記ID: {marker_id}")
        print(f"ArUco位置向量: {aruco_tvec}")
        print(f"ArUco旋轉向量: {aruco_rvec}")
        print(f"機器人姿態: {robot_pose}")
        
        # 轉換為變換矩陣
        T_camera_marker = self.aruco_to_transformation_matrix(aruco_tvec, aruco_rvec)
        T_base_gripper = self.pose_to_transformation_matrix(robot_pose)
        
        print("\n相機到標記的變換矩陣:")
        print(T_camera_marker)
        print("\n基座到夾爪的變換矩陣:")
        print(T_base_gripper)
        
        # 儲存處理後的數據
        processed_data = {
            'marker_id': marker_id,
            'T_camera_marker': T_camera_marker,
            'T_base_gripper': T_base_gripper,
            'original_data': data
        }
        
        return processed_data
    
    def add_calibration_point(self, json_data):
        """
        添加標定點到數據集
        """
        processed_data = self.process_calibration_data(json_data)
        self.calibration_data.append(processed_data)
        print(f"已添加第 {len(self.calibration_data)} 個標定點")
    
    def prepare_opencv_calibration_data(self):
        """
        準備OpenCV手眼標定所需的數據格式
        """
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        
        for data in self.calibration_data:
            # 從基座到夾爪的變換（需要取逆得到夾爪到基座）
            T_base_gripper = data['T_base_gripper']
            T_gripper_base = np.linalg.inv(T_base_gripper)
            
            R_gripper2base.append(T_gripper_base[:3, :3])
            t_gripper2base.append(T_gripper_base[:3, 3])
            
            # 從相機到標記的變換
            T_camera_marker = data['T_camera_marker']
            R_target2cam.append(T_camera_marker[:3, :3])
            t_target2cam.append(T_camera_marker[:3, 3])
        
        return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
    
    def perform_hand_eye_calibration(self, method=cv2.CALIB_HAND_EYE_TSAI):
        """
        執行手眼標定
        """
        if len(self.calibration_data) < 3:
            print("警告：標定點數量少於3個，可能影響標定精度")
        
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam = self.prepare_opencv_calibration_data()
        
        # 執行手眼標定
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=method
        )
        
        # 組成變換矩陣
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
                
        print("\n手眼標定結果 - 相機到夾爪的變換矩陣 (T_camera_gripper):")
        print("T_camera_gripper = np.array([")
        for row in T_cam2gripper:
            formatted_row = ", ".join([f"{val: .8e}" for val in row])
            print(f" [{formatted_row}],")
        print("])")

        
        return T_cam2gripper
    
    def load_multiple_json_files(self, file_paths):
        """
        載入多個JSON標定文件
        """
        loaded_count = 0
        failed_files = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.add_calibration_point(data)
                loaded_count += 1
                print(f"成功載入: {file_path}")
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"載入失敗: {file_path} - {e}")
        
        print(f"\n載入完成: {loaded_count} 個文件成功，{len(failed_files)} 個文件失敗")
        return loaded_count, failed_files

    def load_json_files_from_directory(self, directory_path, pattern="handeye_record_*.json"):
        """
        從目錄載入所有符合模式的JSON文件
        """
        import glob
        import os
        
        search_pattern = os.path.join(directory_path, pattern)
        json_files = glob.glob(search_pattern)
        
        if not json_files:
            print(f"在目錄 {directory_path} 中找不到符合模式 {pattern} 的文件")
            return 0, []
        
        print(f"找到 {len(json_files)} 個JSON文件")
        return self.load_multiple_json_files(json_files)
        """
        保存標定結果
        """
        # 提取旋轉和平移
        rotation_matrix = T_cam2gripper[:3, :3]
        translation = T_cam2gripper[:3, 3]
        
        # 轉換為歐拉角
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        
        result = {
            'transformation_matrix': T_cam2gripper.tolist(),
            'translation': translation.tolist(),
            'rotation_matrix': rotation_matrix.tolist(),
            'euler_angles_deg': euler_angles.tolist(),
            'quaternion': rotation.as_quat().tolist()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"標定結果已保存到: {filename}")

    def load_multiple_json_files(self, file_paths):
        """
        載入多個JSON標定文件
        """
        loaded_count = 0
        failed_files = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.add_calibration_point(data)
                loaded_count += 1
                print(f"成功載入: {file_path}")
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"載入失敗: {file_path} - {e}")
        
        print(f"\n載入完成: {loaded_count} 個文件成功，{len(failed_files)} 個文件失敗")
        return loaded_count, failed_files

    def load_json_files_from_directory(self, directory_path, pattern="handeye_record_*.json"):
        """
        從目錄載入所有符合模式的JSON文件
        """
        import glob
        import os
        
        search_pattern = os.path.join(directory_path, pattern)
        json_files = glob.glob(search_pattern)
        
        if not json_files:
            print(f"在目錄 {directory_path} 中找不到符合模式 {pattern} 的文件")
            return 0, []
        
        print(f"找到 {len(json_files)} 個JSON文件")
        return self.load_multiple_json_files(json_files)

# 使用範例
if __name__ == "__main__":
    # 創建處理器
    processor = HandEyeCalibrationProcessor()
    
    # 從文件讀取包含多組數據的JSON文件
    json_file_path = r"C:\Users\admin\Desktop\robottest\c2h_trans\handeye_records\handeye_record_20250728_143004.json"
    print(f"正在讀取文件: {json_file_path}")
    loaded_count = processor.load_json_file_with_multiple_records(json_file_path)
    
    if loaded_count > 0:
        print(f"\n=== 數據載入完成 ===")
        print(f"總共載入 {loaded_count} 組標定數據")
        
        # 顯示數據摘要
        print(f"標記ID範圍: {[data['original_data']['marker_id'] for data in processor.calibration_data[:5]]}...")
        
        if loaded_count >= 3:
            print("\n=== 執行手眼標定 ===")
            T_result = processor.perform_hand_eye_calibration()
            
            # 保存結果，文件名包含時間戳
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f'hand_eye_calibration_result_{timestamp}.json'
            #processor.save_calibration_result(T_result, result_filename)
        else:
            print(f"標定點數量不足: {loaded_count} 個，需要至少3個")
    else:
        print("未能載入任何標定數據")