import cv2
import time

print("尝试用OpenCV直接访问奥比中光相机...")

# 通常索引0是默认摄像头，奥比中光可能会出现在索引1,2等
for camera_index in [0, 1, 2, 3]:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # 在Windows上使用DSHOW后端
    if cap.isOpened():
        print(f"成功在索引 {camera_index} 上打开视频设备")
        ret, frame = cap.read()
        if ret:
            print(f"  成功读取一帧，形状: {frame.shape}")
            cv2.imshow('OpenCV Direct Test', frame)
            cv2.waitKey(1000)  # 显示1秒
            cv2.destroyAllWindows()
        else:
            print(f"  警告：无法从该设备读取帧")
        cap.release()
    else:
        print(f"索引 {camera_index} 无可用设备")

print("OpenCV直接访问测试结束。")