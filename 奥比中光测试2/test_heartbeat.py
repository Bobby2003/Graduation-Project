import cv2
import time
import numpy as np
print("程序心跳测试开始...")
last_time = time.time()
frame_count = 0

# 创建一个简单的窗口
cv2.namedWindow("Heartbeat", cv2.WINDOW_NORMAL)

try:
    while True:
        frame_count += 1
        current_time = time.time()

        # 每1秒打印一次心跳（FPS）
        if current_time - last_time >= 1.0:
            fps = frame_count / (current_time - last_time)
            print(f"主循环心跳: {fps:.1f} FPS | 循环次数: {frame_count}")
            frame_count = 0
            last_time = current_time

        # 创建一个纯色图像并显示，模拟UI更新
        dummy_frame = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, f"Heartbeat Test", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Heartbeat", dummy_frame)

        # 关键：检查waitKey的响应
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    print("心跳测试结束。")