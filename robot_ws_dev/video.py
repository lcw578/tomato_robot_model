import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime


def setup_short_range_preset(device):
    """设置RealSense相机的短距离预设"""
    try:
        depth_sensor = device.first_depth_sensor()

        if depth_sensor.supports(rs.option.visual_preset):
            print("Setting Visual Preset to Short Range...")
            depth_sensor.set_option(rs.option.visual_preset, 3)

            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, 100)
                print(f"Laser power set to: {depth_sensor.get_option(rs.option.laser_power)}")

            if depth_sensor.supports(rs.option.receiver_gain):
                depth_sensor.set_option(rs.option.receiver_gain, 18)
                print(f"Receiver gain set to: {depth_sensor.get_option(rs.option.receiver_gain)}")

            return True
        else:
            print("Visual preset not supported by this device")
            return False

    except Exception as e:
        print(f"Error setting short range preset: {e}")
        return False


def main():
    # 初始化相机
    pipeline = rs.pipeline()
    config = rs.config()

    # 只启用彩色流
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # 启动管道
    pipeline_profile = pipeline.start(config)

    # 获取设备并设置短距离预设
    device = pipeline_profile.get_device()
    setup_short_range_preset(device)

    # 创建视频写入器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"color_video_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (1920, 1080))

    print("L515 Color Video Recorder")
    print("Press SPACE to start/stop recording")
    print("Press ESC to exit")

    recording = False
    frame_count = 0

    try:
        while True:
            # 获取帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())

            # 创建用于显示的缩小版本
            display_scale = 0.4
            color_display = cv2.resize(color_image, None, fx=display_scale, fy=display_scale)

            # 添加状态信息
            status = "RECORDING" if recording else "STANDBY"
            status_color = (0, 0, 255) if recording else (0, 255, 0)

            status_text = f"Status: {status} | Frames: {frame_count} | SPACE: Start/Stop | ESC: Exit"
            cv2.putText(color_display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            filename_text = f"File: {video_filename}"
            cv2.putText(color_display, filename_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('L515 Color Video Recorder', color_display)

            # 如果正在录制，写入帧
            if recording:
                video_writer.write(color_image)
                frame_count += 1

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break
            elif key == 32:  # SPACE键 - 开始/停止录制
                if not recording:
                    # 开始录制
                    recording = True
                    frame_count = 0
                    print(f"Started recording: {video_filename}")
                else:
                    # 停止录制
                    recording = False
                    video_writer.release()
                    print(f"Stopped recording. Saved {frame_count} frames to {video_filename}")

                    # 创建新的视频文件准备下一次录制
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"color_video_{timestamp}.avi"
                    video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (1920, 1080))

    finally:
        # 确保释放所有资源
        if recording:
            video_writer.release()
            print(f"Saved {frame_count} frames to {video_filename}")

        pipeline.stop()
        cv2.destroyAllWindows()
        print("Video recording session completed!")


if __name__ == "__main__":
    main()
