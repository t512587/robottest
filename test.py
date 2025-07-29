import numpy as np
from pymycobot.elephantrobot import ElephantRobot, JogMode
import time
import sys # 引入 sys 模組用於退出程式

# 初始化連線
elephant_client = ElephantRobot("192.168.1.159", 5001)
print("嘗試啟動機器人連線...")
try:
    if not elephant_client.start_client():
        print("❌ 連線失敗！請檢查 IP 地址、埠號和機器人網路連線。")
        sys.exit() # 如果連線失敗，則退出程式
    print("✅ 已啟動連線")
except Exception as e:
    print(f"❌ 連線過程中發生錯誤: {e}")
    sys.exit()

# 檢查機器人是否已通電並啟用
print("檢查機器人狀態...")
if not elephant_client.is_power_on():
    print("⚠️ 機器人未通電。正在嘗試啟動機器人電源...")
    elephant_client.start_robot() # 呼叫 start_robot() 也會嘗試啟用
    time.sleep(5) # 等待機器人通電和初始化

if not elephant_client.state_check():
    print("⚠️ 機器人未啟用。正在嘗試啟用機器人...")
    elephant_client.start_robot() # 再次呼叫確保啟用
    time.sleep(2) # 給機器人一點時間響應

if elephant_client.is_collision_detected():
    print("⚠️ 偵測到碰撞。正在嘗試恢復機器人...")
    elephant_client.recover_robot()
    time.sleep(5) # 等待機器人恢復

# 🟢 移動到初始化角度
init_angles = [90, -90, 100, -15, 90, -110]
angle_speed = 500 # 建議將速度設定為 1-100 之間的百分比值
print(f"🟢 正在將機器人移動到初始角度: {init_angles}，速度: {angle_speed}")
elephant_client.write_angles(init_angles, angle_speed)

# 等待手臂完成角度移動
start_time_angles = time.time()
timeout_angles = 30 # 最長等待秒數，確保機器人有足夠時間到達

while True:
    # 獲取當前角度並檢查是否到達目標
    current_angles = elephant_client.get_angles()
    # 確保 is_in_position 判斷的是布林值
    if elephant_client.is_in_position(init_angles, JogMode.JOG_JOINT):
        print("✅ 手臂已抵達初始角度位置")
        break
    elif time.time() - start_time_angles > timeout_angles:
        print("⏰ 初始角度移動超時。機器人可能未能到達目標位置。")
        print(f"  最後已知角度: {current_angles}")
        break
    else:
        # print(f"當前角度: {current_angles}，等待中...") # 可選：用於除錯
        time.sleep(0.5) # 短暫延遲，避免過於頻繁地發送查詢指令

print("\n🎉 角度初始化操作完成")

# 完成所有操作後，記得關閉連線
elephant_client.stop_client()
print("🔌 已關閉連線")