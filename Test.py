"""
R200摄像头直接测试
"""

import cv2
import numpy as np
import time

print("="*60)
print("R200摄像头直接测试")
print("="*60)

# 基于之前的发现，这些是最有可能的索引
test_indices = [0, 1, 2, 700]

for idx in test_indices:
    print(f"\n测试索引 {idx}:")

    # 尝试打开摄像头
    cap = cv2.VideoCapture(idx)

    if not cap.isOpened():
        print(f"  无法打开索引 {idx}")
        continue

    # 获取信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  ✅ 已打开: {width}x{height}")

    # 读取几帧
    frames = []
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    if frames:
        frame = frames[-1]
        print(f"  图像: {frame.shape}, {frame.dtype}")
        print(f"  值范围: {frame.min()} - {frame.max()}")

        # 显示预览
        display_frame = frame.copy()

        # 如果是16位，处理为8位显示
        if frame.dtype == np.uint16:
            print(f"  🎯 发现16位图像！可能是深度图")
            depth_normalized = cv2.normalize(frame, None, 0, 65535, cv2.NORM_MINMAX)
            depth_8bit = np.uint8(depth_normalized / 256)
            display_frame = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        # 调整大小显示
        h, w = display_frame.shape[:2]
        scale = min(400/h, 600/w)
        new_h, new_w = int(h*scale), int(w*scale)
        resized = cv2.resize(display_frame, (new_w, new_h))

        cv2.imshow(f"Index {idx} - {width}x{height}", resized)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    else:
        print("  无法读取帧")

    cap.release()

print("\n测试完成！")
print("\n建议:")
print("1. 索引0: 628x469 - 可能是特殊IR摄像头")
print("2. 索引1: 640x480 - 可能是左IR摄像头")
print("3. 索引2: 640x481 - 可能是右IR摄像头")
print("4. 检查是否有16位图像（深度图）")