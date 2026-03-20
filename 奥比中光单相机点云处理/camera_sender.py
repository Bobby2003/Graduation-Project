#!/usr/bin/env python3
import socket
import numpy as np
import time
from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path

WSL_IP = "127.0.0.1"
PORT = 9999


def main():
    sdk = OrbbecCameraSDK(get_default_sdk_path())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((WSL_IP, PORT))
    print(f"✓ 已连接到 WSL {WSL_IP}:{PORT}")

    try:
        while True:
            depth_frame = sdk.get_depth_frame()
            if depth_frame is not None:
                h, w = depth_frame.shape
                header = f"{h},{w}\n".encode()
                sock.sendall(header)

                data = depth_frame.tobytes()
                sock.sendall(data)
                print(f"发送 {h}x{w} 深度图")
                time.sleep(0.033)
    except KeyboardInterrupt:
        print("停止")
    finally:
        sock.close()
        sdk.close()


if __name__ == "__main__":
    main()