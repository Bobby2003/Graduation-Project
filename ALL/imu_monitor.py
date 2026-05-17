#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import struct
import time
from datetime import datetime

import serial

HEADER = b"\x7E\x23"

def checksum_ok(frame: bytes) -> bool:
    """
    checksum = 从包头1累加到校验位前的值，取最低一个字节
    """
    if len(frame) < 4:
        return False
    calc = sum(frame[:-1]) & 0xFF
    recv = frame[-1]
    return calc == recv

def hexstr(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)

def int16_le(data: bytes, offset: int) -> int:
    return struct.unpack_from("<h", data, offset)[0]

def uint16_le(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]

def float_le(data: bytes, offset: int) -> float:
    """
    协议中四元数、欧拉角、气压计数据为小端 4 字节小数。
    这里按 IEEE754 float32 小端解析。
    """
    return struct.unpack_from("<f", data, offset)[0]

def parse_frame(frame: bytes) -> dict:
    """
    帧格式:
    [0] 0x7E
    [1] 0x23
    [2] length
    [3] func
    [...]
    [last] checksum
    """
    result = {
        "raw": hexstr(frame),
        "valid_checksum": checksum_ok(frame),
        "length": frame[2] if len(frame) >= 3 else None,
        "func": frame[3] if len(frame) >= 4 else None,
        "name": "UNKNOWN",
        "data": {},
    }

    if len(frame) < 4:
        result["name"] = "INVALID_SHORT_FRAME"
        return result

    func = frame[3]

    # 0x01 返回固件版本号
    # 7E 23 08 01 major minor patch checksum
    if func == 0x01 and len(frame) >= 8:
        result["name"] = "固件版本号"
        major = frame[4]
        minor = frame[5]
        patch = frame[6]
        result["data"] = {
            "major": major,
            "minor": minor,
            "patch": patch,
            "version": f"{major}.{minor}.{patch}",
        }

    # 0x81 返回状态
    # 7E 23 07 81 功能 状态 checksum
    elif func == 0x81 and len(frame) >= 7:
        target_func = frame[4]
        status = frame[5]

        target_name_map = {
            0x70: "校准 IMU，陀螺仪和加速度计",
            0x71: "校准磁力计",
            0x73: "校准温度",
        }

        result["name"] = "状态返回"
        result["data"] = {
            "target_func": f"0x{target_func:02X}",
            "target_name": target_name_map.get(target_func, "未知功能"),
            "status": status,
            "status_text": "成功" if status == 1 else "失败",
        }

    # 0x04 返回原始数据
    # length 0x17 = 23 bytes
    elif func == 0x04 and len(frame) >= 23:
        result["name"] = "原始数据"

        ax_raw = int16_le(frame, 4)
        ay_raw = int16_le(frame, 6)
        az_raw = int16_le(frame, 8)

        gx_raw = int16_le(frame, 10)
        gy_raw = int16_le(frame, 12)
        gz_raw = int16_le(frame, 14)

        mx_raw = int16_le(frame, 16)
        my_raw = int16_le(frame, 18)
        mz_raw = int16_le(frame, 20)

        accel_scale = 16.0 / 32767.0
        gyro_scale = (2000.0 / 32767.0) * (math.pi / 180.0)
        mag_scale = 800.0 / 32767.0

        result["data"] = {
            "accel_raw": {
                "x": ax_raw,
                "y": ay_raw,
                "z": az_raw,
            },
            "gyro_raw": {
                "x": gx_raw,
                "y": gy_raw,
                "z": gz_raw,
            },
            "mag_raw": {
                "x": mx_raw,
                "y": my_raw,
                "z": mz_raw,
            },
            "accel_g": {
                "x": ax_raw * accel_scale,
                "y": ay_raw * accel_scale,
                "z": az_raw * accel_scale,
            },
            "gyro_rad_s": {
                "x": gx_raw * gyro_scale,
                "y": gy_raw * gyro_scale,
                "z": gz_raw * gyro_scale,
            },
            "gyro_deg_s": {
                "x": gx_raw * 2000.0 / 32767.0,
                "y": gy_raw * 2000.0 / 32767.0,
                "z": gz_raw * 2000.0 / 32767.0,
            },
            "mag": {
                "x": mx_raw * mag_scale,
                "y": my_raw * mag_scale,
                "z": mz_raw * mag_scale,
            },
        }

    # 0x16 返回四元数
    # 7E 23 15 16 Q0[4] Q1[4] Q2[4] Q3[4] checksum
    elif func == 0x16 and len(frame) >= 21:
        result["name"] = "四元数"

        q0 = float_le(frame, 4)
        q1 = float_le(frame, 8)
        q2 = float_le(frame, 12)
        q3 = float_le(frame, 16)

        result["data"] = {
            "w": q0,
            "x": q1,
            "y": q2,
            "z": q3,
        }

    # 0x26 返回欧拉角
    # 7E 23 11 26 ROLL[4] PITCH[4] YAW[4] checksum
    elif func == 0x26 and len(frame) >= 17:
        result["name"] = "欧拉角"

        roll = float_le(frame, 4)
        pitch = float_le(frame, 8)
        yaw = float_le(frame, 12)

        result["data"] = {
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
        }

    # 0x32 返回气压计数据，十轴才有
    # 7E 23 15 32 height[4] temperature[4] pressure[4] pressure_contrast[4] checksum
    elif func == 0x32 and len(frame) >= 21:
        result["name"] = "气压计数据"

        height = float_le(frame, 4)
        temperature = float_le(frame, 8)
        pressure = float_le(frame, 12)
        pressure_contrast = float_le(frame, 16)

        result["data"] = {
            "height_m": height,
            "temperature_c": temperature,
            "pressure_pa": pressure,
            "pressure_contrast_pa": pressure_contrast,
        }

    else:
        result["name"] = f"未知帧/未实现功能字 0x{func:02X}"

    return result

def format_output(parsed: dict, show_raw: bool = False) -> str:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    ok = "OK" if parsed["valid_checksum"] else "BAD_CHECKSUM"

    name = parsed["name"]
    func = parsed["func"]
    func_str = f"0x{func:02X}" if func is not None else "None"

    lines = [
        f"[{ts}] [{ok}] {name} func={func_str} len={parsed['length']}"
    ]

    data = parsed.get("data", {})

    if name == "固件版本号":
        lines.append(f"  version: {data.get('version')}")

    elif name == "状态返回":
        lines.append(
            f"  target: {data.get('target_name')} "
            f"({data.get('target_func')}), status: {data.get('status_text')}({data.get('status')})"
        )

    elif name == "原始数据":
        a = data["accel_g"]
        g_rad = data["gyro_rad_s"]
        g_deg = data["gyro_deg_s"]
        m = data["mag"]

        lines.append(
            f"  ACCEL[g]    x={a['x']:+.6f}, y={a['y']:+.6f}, z={a['z']:+.6f}"
        )
        lines.append(
            f"  GYRO[rad/s] x={g_rad['x']:+.6f}, y={g_rad['y']:+.6f}, z={g_rad['z']:+.6f}"
        )
        lines.append(
            f"  GYRO[deg/s] x={g_deg['x']:+.3f}, y={g_deg['y']:+.3f}, z={g_deg['z']:+.3f}"
        )
        lines.append(
            f"  MAG         x={m['x']:+.6f}, y={m['y']:+.6f}, z={m['z']:+.6f}"
        )

    elif name == "四元数":
        lines.append(
            f"  QUAT        w={data['w']:+.8f}, x={data['x']:+.8f}, "
            f"y={data['y']:+.8f}, z={data['z']:+.8f}"
        )

    elif name == "欧拉角":
        lines.append(
            f"  EULER[rad]  roll={data['roll_rad']:+.8f}, "
            f"pitch={data['pitch_rad']:+.8f}, yaw={data['yaw_rad']:+.8f}"
        )
        lines.append(
            f"  EULER[deg]  roll={data['roll_deg']:+.3f}, "
            f"pitch={data['pitch_deg']:+.3f}, yaw={data['yaw_deg']:+.3f}"
        )

    elif name == "气压计数据":
        lines.append(
            f"  BARO        height={data['height_m']:+.3f} m, "
            f"temp={data['temperature_c']:+.3f} °C"
        )
        lines.append(
            f"              pressure={data['pressure_pa']:.3f} Pa, "
            f"contrast={data['pressure_contrast_pa']:.3f} Pa"
        )

    else:
        lines.append(f"  data: {data}")

    if show_raw:
        lines.append(f"  raw: {parsed['raw']}")

    return "\n".join(lines)

def make_cmd(func: int, payload: bytes) -> bytes:
    """
    上位机发送命令:
    7E 23 length func payload checksum

    length = 整帧长度 = checksum 下标 + 1
    """
    length = 2 + 1 + 1 + len(payload) + 1
    frame_without_checksum = bytes([0x7E, 0x23, length, func]) + payload
    chk = sum(frame_without_checksum) & 0xFF
    return frame_without_checksum + bytes([chk])

def cmd_request_version() -> bytes:
    # 协议固定：7E 23 07 80 01 00 29
    return make_cmd(0x80, bytes([0x01, 0x00]))

def cmd_set_rate(rate_hz: int) -> bytes:
    # 设置数据输出频率：7E 23 07 60 XX 5F checksum
    if not 10 <= rate_hz <= 100:
        raise ValueError("输出频率范围必须是 10~100 Hz")
    return make_cmd(0x60, bytes([rate_hz & 0xFF, 0x5F]))

def cmd_set_algo(algo: int) -> bytes:
    # 设置算法类型：6 或 9
    if algo not in (6, 9):
        raise ValueError("算法类型只能是 6 或 9")
    return make_cmd(0x61, bytes([algo, 0x5F]))

def cmd_calib_imu(start: bool) -> bytes:
    # 校准 IMU：参数1=1开始，0清除；参数2=0x5F
    return make_cmd(0x70, bytes([0x01 if start else 0x00, 0x5F]))

def cmd_calib_mag(start: bool) -> bytes:
    # 校准磁力计
    return make_cmd(0x71, bytes([0x01 if start else 0x00, 0x5F]))

def cmd_calib_temp(temp_c: float) -> bytes:
    # 温度校准：实际温度 * 100，uint16 小端
    t100 = int(round(temp_c * 100))
    payload = struct.pack("<H", t100) + bytes([0x5F])
    return make_cmd(0x73, payload)

def cmd_reset_user_data() -> bytes:
    # 协议固定：7E 23 07 A0 01 5F A8
    return make_cmd(0xA0, bytes([0x01, 0x5F]))

def read_frames(ser: serial.Serial):
    """
    按协议实时读取帧：
    - 找包头 7E 23
    - 读取 length
    - 再读取剩余 length - 3 字节
    """
    buffer = bytearray()

    while True:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buffer.extend(chunk)

        while True:
            # 找包头
            idx = buffer.find(HEADER)
            if idx < 0:
                # 保留最后 1 字节，防止它是 0x7E
                if len(buffer) > 1:
                    del buffer[:-1]
                break

            # 丢弃包头前噪声
            if idx > 0:
                del buffer[:idx]

            # 至少需要 3 字节才能知道长度
            if len(buffer) < 3:
                break

            length = buffer[2]

            # 基本合法性检查
            if length < 5 or length > 255:
                # 长度异常，丢弃第一个字节重新同步
                del buffer[0]
                continue

            # 等待完整帧
            if len(buffer) < length:
                break

            frame = bytes(buffer[:length])
            del buffer[:length]

            yield frame

def main():
    parser = argparse.ArgumentParser(description="IMU 串口实时测试脚本")
    parser.add_argument("-p", "--port", required=True, help="串口号，例如 COM3 或 /dev/ttyUSB0")
    parser.add_argument("-b", "--baud", type=int, default=115200, help="波特率，默认 115200")
    parser.add_argument("--timeout", type=float, default=0.1, help="串口读取超时时间")
    parser.add_argument("--raw", action="store_true", help="显示原始 HEX 帧")
    parser.add_argument("--request-version", action="store_true", help="启动后请求固件版本号")
    parser.add_argument("--set-rate", type=int, help="设置输出频率，范围 10~100 Hz")
    parser.add_argument("--set-algo", type=int, choices=[6, 9], help="设置算法类型，6 或 9")
    parser.add_argument("--calib-imu", choices=["start", "clear"], help="校准 IMU：start 或 clear")
    parser.add_argument("--calib-mag", choices=["start", "clear"], help="校准磁力计：start 或 clear")
    parser.add_argument("--calib-temp", type=float, help="校准温度，例如 25.30")
    parser.add_argument("--reset-user-data", action="store_true", help="重置用户数据")
    args = parser.parse_args()

    print(f"Opening serial port: {args.port}, baud={args.baud}")

    with serial.Serial(
        port=args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=args.timeout,
    ) as ser:
        time.sleep(0.2)
        ser.reset_input_buffer()

        # 可选发送命令
        if args.request_version:
            cmd = cmd_request_version()
            print(f"TX request version: {hexstr(cmd)}")
            ser.write(cmd)

        if args.set_rate is not None:
            cmd = cmd_set_rate(args.set_rate)
            print(f"TX set rate {args.set_rate}Hz: {hexstr(cmd)}")
            ser.write(cmd)

        if args.set_algo is not None:
            cmd = cmd_set_algo(args.set_algo)
            print(f"TX set algo {args.set_algo}: {hexstr(cmd)}")
            ser.write(cmd)

        if args.calib_imu is not None:
            cmd = cmd_calib_imu(args.calib_imu == "start")
            print(f"TX calib imu {args.calib_imu}: {hexstr(cmd)}")
            ser.write(cmd)

        if args.calib_mag is not None:
            cmd = cmd_calib_mag(args.calib_mag == "start")
            print(f"TX calib mag {args.calib_mag}: {hexstr(cmd)}")
            ser.write(cmd)

        if args.calib_temp is not None:
            cmd = cmd_calib_temp(args.calib_temp)
            print(f"TX calib temp {args.calib_temp}: {hexstr(cmd)}")
            ser.write(cmd)

        if args.reset_user_data:
            cmd = cmd_reset_user_data()
            print(f"TX reset user data: {hexstr(cmd)}")
            ser.write(cmd)

        print("Start monitoring IMU frames. Press Ctrl+C to stop.\n")

        try:
            for frame in read_frames(ser):
                parsed = parse_frame(frame)

                # 清屏并回到左上角，在同一个位置刷新显示
                print("\033[2J\033[H", end="")
                print(format_output(parsed, show_raw=args.raw))
                print("\nPress Ctrl+C to stop.")
        except KeyboardInterrupt:
            print("Stopped.")

if __name__ == "__main__":
    main()