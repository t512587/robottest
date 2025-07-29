import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_transformation_matrix(pose):
    """
    將6DOF姿態轉換為4x4變換矩陣
    pose: [x, y, z, rx, ry, rz] (位置單位:mm，角度單位:度)
    """
    x, y, z, rx, ry, rz = pose
    
    # 位置向量 (此範例不換單位，需確保輸入單位一致)
    translation = np.array([x, y, z])
    
    # 歐拉角轉旋轉矩陣 (xyz順序，角度轉弧度)
    rotation = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    rotation_matrix = rotation.as_matrix()
    
    # 組成4x4變換矩陣
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    
    return T

def is_identity(matrix, tol=1e-9):
    """
    判斷矩陣是否為單位矩陣（允許誤差tol）
    """
    return np.allclose(matrix, np.eye(matrix.shape[0]), atol=tol)

if __name__ == "__main__":
    # 假設末端在基座座標系下的位姿（位置單位mm，角度度）
    robot_pose = [100, 200, 300, 30, -45, 60]  # 範例數據
    
    # 轉成基座到末端的變換矩陣
    T_base_gripper = pose_to_transformation_matrix(robot_pose)
    print("基座到末端變換矩陣 T_base_gripper:")
    print(T_base_gripper)
    
    # 求逆，得到末端到基座的變換矩陣
    T_gripper_base = np.linalg.inv(T_base_gripper)
    print("\n末端到基座變換矩陣 T_gripper_base:")
    print(T_gripper_base)
    
    # 驗證兩者互為逆矩陣
    identity_check_1 = T_base_gripper @ T_gripper_base
    identity_check_2 = T_gripper_base @ T_base_gripper
    
    print("\nT_base_gripper @ T_gripper_base 是否為單位矩陣？", is_identity(identity_check_1))
    print(identity_check_1)
    
    print("\nT_gripper_base @ T_base_gripper 是否為單位矩陣？", is_identity(identity_check_2))
    print(identity_check_2)
