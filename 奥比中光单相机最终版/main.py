"""
彩色3D重建系统 - 程序入口
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from color_3d_reconstructor import main

if __name__ == "__main__":
    # 设置环境变量
    default_sdk_path = r"H:\School assessment\Design\奥比中光Win64-Release\sdk\libs"
    if os.path.exists(default_sdk_path):
        drivers_dir = os.path.join(default_sdk_path, "OpenNI2", "Drivers")
        if os.path.exists(drivers_dir):
            os.environ['PATH'] = drivers_dir + ';' + os.environ['PATH']
            print(f"✅ 已设置环境变量: {drivers_dir}")

    # 运行主程序
    main()