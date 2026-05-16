import cv2
import socket
import time
import numpy as np
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path

WSL_IP = "172.17.122.201"
PORT = 9999

def main():
    sdk = OrbbecCameraSDK(get_default_sdk_path())
    if not sdk.initialize(): print("初始化失败"); return
    if not sdk.open_device(): sdk.cleanup(); return
    if not sdk.create_stream(sdk.ONI_SENSOR_DEPTH): sdk.cleanup(); return
    if not sdk.start_stream(sdk.depth_stream_handle): sdk.cleanup(); return

    # 用 OpenCV 打开彩色流，index 根据你实际情况调整（0 或 1 或 2）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("OpenCV 打开彩色流失败"); sdk.cleanup(); return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((WSL_IP, PORT))
    print(f"✓ 已连接到 WSL {WSL_IP}:{PORT}")

    try:
        while True:
            depth_result = sdk.capture_depth_frame(timeout=1000)
            ret, color_bgr = cap.read()

            print(f"depth={depth_result is not None}, color={ret}")

            if depth_result and ret:
                depth_data, _ = depth_result
                # OpenCV 读出来是 BGR，转 RGB
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

                dh, dw = depth_data.shape
                ch, cw, _ = color_rgb.shape

                header = f"{dh},{dw},{ch},{cw}\n".encode()
                sock.sendall(header)
                sock.sendall(depth_data.tobytes())
                sock.sendall(color_rgb.tobytes())
                print(f"发送 depth={dh}x{dw} color={ch}x{cw}")

            time.sleep(0.033)
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        sock.close()
        sdk.cleanup()

if __name__ == "__main__":
    main()