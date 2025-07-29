from pymycobot import ElephantRobot
import time

def check_elephant_robot():
    # 建立連線
    elephant = ElephantRobot("192.168.1.159", 5001)
    elephant.start_client()
    print("✅ 已啟動連線")

    # 嘗試上電（使用 _power_on）
    elephant._power_on()
    time.sleep(3)  # 增加等待時間，確保硬體響應

    # 檢查是否成功上電
    power_state = elephant.is_power_on()
    print(f"🔌 is_power_on(): {power_state}")
    if power_state:
        print("⚡ 已上電成功")
    else:
        print("⚠️ 上電失敗（但可能只是無法偵測）")

    # 啟動機器人
    elephant.start_robot()
    print("▶️ 已啟動機器人")

    # 額外測試狀態：角度與位置
    try:
        angles = elephant.get_angles()
        print(f"📐 當前角度: {angles}")
        coords = elephant.get_coords()
        print(f"📍 當前位置: {coords}")
    except Exception as e:
        print(f"❌ 無法取得角度/位置: {e}")

    # 狀態檢查（True 表示連線穩定）
    state = elephant.state_check()
    print(f"📋 Robot 連線狀態: {state}")

    # 是否正在運行指令
    is_running = elephant.check_running()
    print(f"🏃 是否正在運行: {is_running}")

    # 測試下電
    print("🛑 準備下電...")
    elephant._power_off()
    time.sleep(2)
    print(f"🔌 下電狀態 is_power_on(): {elephant.is_power_on()}")

if __name__ == "__main__":
    check_elephant_robot()
