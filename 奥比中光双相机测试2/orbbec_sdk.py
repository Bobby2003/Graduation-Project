"""
奥比中光相机SDK接口层 - 多实例增强版
支持同时打开多个相机并独立获取流数据
"""
import ctypes
import numpy as np
import os
import sys
from typing import Optional, Tuple, Dict, Any

class OrbbecCameraSDK:
    # 常量定义
    ONI_MAX_STR = 256
    ONI_STATUS_OK = 0
    ONI_SENSOR_DEPTH = 3
    ONI_SENSOR_COLOR = 2
    ONI_PIXEL_FORMAT_DEPTH_1_MM = 100
    ONI_PIXEL_FORMAT_RGB888 = 200
    ONI_API_VERSION = 2000

    _sdk_initialized = False # 类变量，确保oniInitialize只调用一次

    def __init__(self, sdk_path: Optional[str] = None):
        self.sdk_path = sdk_path
        self.lib = None
        self.device_handle = ctypes.c_void_p()
        self.depth_stream_handle = ctypes.c_void_p()
        self.is_device_open = False
        self._define_structures()
        self._load_library()

    def _define_structures(self):
        class OniDeviceInfo(ctypes.Structure):
            _fields_ = [("uri", ctypes.c_char * 256), ("vendor", ctypes.c_char * 256),
                        ("name", ctypes.c_char * 256), ("serialNumber", ctypes.c_char * 256),
                        ("usbVendorId", ctypes.c_uint16), ("usbProductId", ctypes.c_uint16)]
        class OniVideoMode(ctypes.Structure):
            _fields_ = [("pixelFormat", ctypes.c_int), ("resolutionX", ctypes.c_int),
                        ("resolutionY", ctypes.c_int), ("fps", ctypes.c_int)]
        class OniFrame(ctypes.Structure):
            _fields_ = [("dataSize", ctypes.c_int), ("data", ctypes.c_void_p),
                        ("sensorType", ctypes.c_int), ("timestamp", ctypes.c_uint64),
                        ("frameIndex", ctypes.c_int), ("width", ctypes.c_int),
                        ("height", ctypes.c_int), ("videoMode", OniVideoMode),
                        ("croppingEnabled", ctypes.c_int), ("cropOriginX", ctypes.c_int),
                        ("cropOriginY", ctypes.c_int), ("stride", ctypes.c_int)]
        self.OniDeviceInfo, self.OniFrame = OniDeviceInfo, OniFrame

    def _load_library(self):
        dll_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs\OpenNI2.dll"
        if not os.path.exists(dll_path): dll_path = "OpenNI2.dll"
        self.lib = ctypes.CDLL(dll_path)
        # 定义核心API函数原型
        self.lib.oniInitialize.argtypes = [ctypes.c_int]
        self.lib.oniGetDeviceList.argtypes = [ctypes.POINTER(ctypes.POINTER(self.OniDeviceInfo)), ctypes.POINTER(ctypes.c_int)]
        self.lib.oniDeviceOpen.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.oniDeviceCreateStream.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.oniStreamReadFrame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(self.OniFrame))]

    def initialize(self):
        if not OrbbecCameraSDK._sdk_initialized:
            res = self.lib.oniInitialize(self.ONI_API_VERSION)
            if res == 0: OrbbecCameraSDK._sdk_initialized = True
            return res == 0
        return True

    def get_devices(self):
        devs_ptr = ctypes.POINTER(self.OniDeviceInfo)()
        count = ctypes.c_int(0)
        self.lib.oniGetDeviceList(ctypes.byref(devs_ptr), ctypes.byref(count))
        devices = []
        for i in range(count.value):
            devices.append({'uri': devs_ptr[i].uri, 'name': devs_ptr[i].name.decode()})
        return devices

    def open(self, uri):
        res = self.lib.oniDeviceOpen(uri, ctypes.byref(self.device_handle))
        if res == 0:
            self.is_device_open = True
            # 开启深度流
            self.lib.oniDeviceCreateStream(self.device_handle, self.ONI_SENSOR_DEPTH, ctypes.byref(self.depth_stream_handle))
            self.lib.oniStreamStart(self.depth_stream_handle)
        return res == 0

    def get_depth(self):
        frame_ptr = ctypes.POINTER(self.OniFrame)()
        res = self.lib.oniStreamReadFrame(self.depth_stream_handle, ctypes.byref(frame_ptr))
        if res == 0:
            frame = frame_ptr.contents
            data = ctypes.string_at(frame.data, frame.dataSize)
            return np.frombuffer(data, dtype=np.uint16).reshape(frame.height, frame.width).copy()
        return None

    def close(self):
        if self.is_device_open:
            self.lib.oniDeviceClose(self.device_handle)

def get_default_sdk_path():
    return r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"