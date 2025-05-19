import multiprocessing as mp
import time
import sys
import signal
import cv2
import numpy as np
import pickle
import traceback

from dashboard import TruckDashboard
from utils import change_window
import window
import grabscreen
from log import log
from control_handler import process

# Event to control running state
running_event = mp.Event()


def signal_handler(sig, frame):
    log.debug("Gracefully exiting...")
    running_event.clear()
    sys.exit(0)


def wait_for_game_window(running_event):
    while running_event.is_set():
        frame = grabscreen.grab_screen()
        if frame is not None and window.set_windows_offset(frame):
            log.debug("Game window detected and offsets set!")
            return True
        time.sleep(1)
    return False


def main():
    try:
        # 初始化系统
        initialize_system()

        # 创建共享数据结构
        shared_data, lane_status_buffer, lane_status_shape = setup_shared_data()

        # 创建仪表盘
        dashboard = TruckDashboard()

        # 启动大脑处理进程
        brain_process = start_brain_process(shared_data, lane_status_buffer, lane_status_shape)

        # 主循环处理仪表盘数据
        run_dashboard_loop(dashboard, shared_data, lane_status_buffer, lane_status_shape)

    except KeyboardInterrupt:
        log.debug("Main process: 捕获到键盘中断信号...")
        shutdown_system(brain_process)
    except Exception as e:
        log.error(f"主进程发生错误: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


def initialize_system():
    """初始化系统组件"""
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)

    # 初始化摄像头
    grabscreen.init_camera(target_fps=30)

    # 校正窗口
    change_window.correction_window()

    # 验证窗口分辨率
    if not change_window.check_window_resolution_same(window.game_width, window.game_height):
        raise ValueError(
            f"游戏分辨率和配置game_width({window.game_width}), game_height({window.game_height})不一致，请到window.py中修改"
        )

    # 设置运行状态
    running_event.set()

    # 等待游戏窗口
    if not wait_for_game_window(running_event):
        log.debug("未能检测到游戏窗口。")
        return False

    return True


def setup_shared_data():
    """设置进程间共享数据结构"""
    # 创建Manager用于进程间共享数据
    manager = mp.Manager()
    shared_data = manager.dict()

    # 初始化共享数据
    shared_data.update({
        "speed": 0,
        "distance": 0,
        "time": 0,
        "fuel": 100.0,
        "gear": 0,
        "setspeed": 0,
        "user_steer": 0,
        "game_steer": 0,
        "lane_status": None,
        # "car_detect": None,
        "traffic_detect": None,
        "detect_frame": None
    })

    # 创建共享内存用于传递大型数据
    lane_status_shape = (1200, 600, 3)  # 根据实际情况调整尺寸
    buffer_size = int(np.prod(lane_status_shape) * 4)  # 4字节浮点数
    lane_status_buffer = mp.Array('B', buffer_size)

    return shared_data, lane_status_buffer, lane_status_shape


def start_brain_process(shared_data, lane_status_buffer, lane_status_shape):
    """启动大脑处理进程"""
    brain_process = mp.Process(
        target=process,
        args=(shared_data, lane_status_buffer, lane_status_shape, running_event)
    )
    brain_process.start()
    return brain_process


def run_dashboard_loop(dashboard, shared_data, lane_status_buffer, lane_status_shape):
    """运行仪表盘主循环"""
    update_interval = 0.1
    last_update_time = time.time()

    while running_event.is_set():
        current_time = time.time()

        # 只在间隔时间到达后更新
        if current_time - last_update_time >= update_interval:
            # 更新仪表盘数据
            dashboard.update_data()

            # 获取并更新共享数据
            update_shared_data(dashboard, shared_data, lane_status_buffer, lane_status_shape)

            # 更新最后更新时间
            last_update_time = current_time
        else:
            # 小睡眠以防止CPU占用过高
            time.sleep(0.01)


def update_shared_data(dashboard, shared_data, lane_status_buffer, lane_status_shape):
    """更新共享数据结构"""
    # 获取仪表盘存储的数据
    status = dashboard.get_stored_data()

    # 更新基本数据
    update_basic_data(shared_data, status)

    # 更新车道状态数据
    update_lane_status(shared_data, status, lane_status_buffer, lane_status_shape)


def update_basic_data(shared_data, status):
    """更新基本共享数据"""
    basic_data = {
        "speed": status["speed"],
        "setspeed": status["setspeed"],
        "distance": status["distance"],
        "time": status["time"],
        "fuel": status["fuel"],
        "gear": status["gear"],
        "user_steer": status["user_steer"],
        "game_steer": status["game_steer"],
        # "car_detect": pickle.dumps(status["car_detect"]) if status["car_detect"] else None,
        "traffic_detect": pickle.dumps(status["traffic_detect"]) if status["traffic_detect"] else None,
        "detect_frame": pickle.dumps(status["detect_frame"]) if status["detect_frame"] else None,
    }
    shared_data.update(basic_data)


def update_lane_status(shared_data, status, lane_status_buffer, lane_status_shape):
    """更新车道状态数据"""
    if status["lane_status"] is None:
        shared_data["lane_status"] = None
        return

    # 处理numpy数组类型的车道状态
    if isinstance(status["lane_status"], np.ndarray):
        lane_status_array = np.frombuffer(lane_status_buffer, dtype=np.float32).reshape(lane_status_shape)
        lane_status = status["lane_status"]

        # 处理形状不匹配的情况
        if lane_status.shape != lane_status_shape:
            copy_with_shape_handling(lane_status, lane_status_array, lane_status_shape)
        else:
            lane_status_array[:] = lane_status

        shared_data["lane_status"] = True  # 标记已更新
    else:
        # 非numpy数组类型的处理
        try:
            shared_data["lane_status"] = pickle.dumps(status["lane_status"])
        except Exception as e:
            log.error(f"序列化lane_status时出错: {e}")
            shared_data["lane_status"] = None


def copy_with_shape_handling(source, target_array, target_shape):
    """处理不同形状的数组复制"""
    log.error(f"车道状态形状不匹配: {source.shape} vs {target_shape}")

    if np.prod(source.shape) <= np.prod(target_shape):
        # 源数据小于目标缓冲区
        target_array.fill(0)
        min_dims = [min(source.shape[i], target_shape[i]) for i in
                    range(min(len(source.shape), len(target_shape)))]

        # 根据维度数量复制数据
        if len(min_dims) == 3:
            target_array[:min_dims[0], :min_dims[1], :min_dims[2]] = source[:min_dims[0], :min_dims[1], :min_dims[2]]
        elif len(min_dims) == 2:
            target_array[:min_dims[0], :min_dims[1], 0] = source[:min_dims[0], :min_dims[1]]
        elif len(min_dims) == 1:
            target_array[:min_dims[0], 0, 0] = source[:min_dims[0]]
    else:
        # 源数据大于目标缓冲区，需要截断
        slices = tuple(slice(0, dim) for dim in target_shape)
        target_array[:] = source[slices]


def shutdown_system(brain_process):
    """关闭系统并清理资源"""
    log.debug("主进程：正在终止子进程...")
    running_event.clear()
    brain_process.terminate()
    brain_process.join()
    log.debug("主进程：正在退出。")


if __name__ == '__main__':
    main()