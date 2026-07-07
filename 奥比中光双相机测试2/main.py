"""
双相机水平融合建模系统 - 启动入口
"""
from dual_reconstructor import DualCameraReconstructor
import os
import sys

if __name__ == "__main__":
    # 环境准备
    sdk_libs = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"
    if os.path.exists(sdk_libs):
        os.environ['PATH'] = os.path.join(sdk_libs, "OpenNI2", "Drivers") + ';' + os.environ['PATH']

    # 启动重建
    # 注意：如果画面重合度不高，可以增加迭代次数
    reconstructor = DualCameraReconstructor()
    reconstructor.run(max_frames=30)