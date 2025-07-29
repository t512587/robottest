import numpy as np

def invert_transform(T):
    """反轉一個4x4齊次變換矩陣"""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def transform_point(T, P):
    """
    用4x4變換矩陣 T 轉換齊次座標點 P
    P 是3x1向量 (非齊次)
    """
    P_h = np.ones(4)
    P_h[:3] = P
    P_transformed = T @ P_h
    return P_transformed[:3]

# 你給的相機到夾爪的變換矩陣 (T_camera_gripper)
T_camera_gripper = np.array([
    [ 2.90836209e-01,  7.64953618e-02, -9.53709998e-01, -5.37946158e+01],
    [ 6.43395326e-03,  9.96619855e-01,  8.18991341e-02,  4.52595294e+01],
    [ 9.56751224e-01, -2.99553592e-02,  2.89360973e-01,  2.20382370e+01],
    [ 0.0,             0.0,             0.0,             1.0]
])

# 範例：手臂基座到夾爪的變換矩陣（請換成你手臂FK算出來的矩陣）
T_base_gripper = np.array([
    [1, 0, 0, 235],  # 假設夾爪在基座座標的 x=300 mm
    [0, 1, 0, -0.5],
    [0, 0, 1, 144],  # z=200 mm
    [0, 0, 0, 1]
])

# 物體在相機座標系的位置（以 mm 為單位）
P_camera = np.array([95.7, 11, 421])

# 計算相機到夾爪的反變換 (夾爪到相機)
T_gripper_camera = invert_transform(T_camera_gripper)

# 計算物體在基座座標系的位置：
# P_base = T_base_gripper * T_gripper_camera * P_camera
P_base = transform_point(T_base_gripper @ T_gripper_camera, P_camera)

print("物體在機械手臂基座座標系的位置:", P_base)
