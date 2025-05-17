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
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize camera
    grabscreen.init_camera(target_fps=30)

    change_window.correction_window()

    if change_window.check_window_resolution_same(window.game_width, window.game_height) == False:
        raise ValueError(
            f"游戏分辨率和配置game_width({window.game_width}), game_height({window.game_height})不一致，请到window.py中修改"
        )

    running_event.set()

    # Wait for game window
    if not wait_for_game_window(running_event):
        log.debug("Failed to detect game window.")
        return

    # 创建一个Manager用于进程间共享数据
    manager = mp.Manager()
    shared_data = manager.dict()

    # 初始化共享数据
    shared_data.update({
        "speed": 0,
        "distance": 0,
        "time": 0,
        "fuel": 100.0,
        "lane_status": None,
        "classes": [],
        "frame_updated": False  # 标记是否有新的帧数据
    })

    # 创建共享内存用于传递大型数据（如lane_status可能是numpy数组）
    lane_status_shape = (720, 1280, 3)  # 假设的最大尺寸，根据实际情况调整
    # 修复: 将numpy.int32转换为Python int
    buffer_size = int(np.prod(lane_status_shape) * 4)  # 4字节浮点数
    lane_status_buffer = mp.Array('B', buffer_size)

    # Create and initialize Dashboard
    dashboard = TruckDashboard()

    # Start child process
    p_brain = mp.Process(target=process, args=(shared_data, lane_status_buffer, lane_status_shape, running_event))
    p_brain.start()

    # Set update interval (0.5 seconds)
    update_interval = 0.5
    last_update_time = time.time()

    try:
        while running_event.is_set():
            current_time = time.time()
            elapsed = current_time - last_update_time

            # Only update when interval has passed
            if elapsed >= update_interval:
                dashboard.update_data()

                # 更新共享数据
                status = dashboard.get_stored_data()

                # 复制基本数据到共享字典
                basic_data = {
                    "speed": status["speed"],
                    "setspeed": status["setspeed"],
                    "distance": status["distance"],
                    "time": status["time"],
                    "fuel": status["fuel"],
                    "gear": status["gear"],
                    "classes": pickle.dumps(status["classes"]) if status["classes"] else [],  # 序列化对象列表
                    "frame_updated": True if status["detect_frame"] is not None else False
                }
                shared_data.update(basic_data)

                # 处理lane_status数据
                if status["lane_status"] is not None:
                    # 如果lane_status是numpy数组，将它复制到共享内存
                    if isinstance(status["lane_status"], np.ndarray):
                        lane_status_array = np.frombuffer(lane_status_buffer, dtype=np.float32).reshape(
                            lane_status_shape)
                        # 确保大小匹配，如果不匹配则调整大小或截断
                        lane_status = status["lane_status"]
                        if lane_status.shape != lane_status_shape:
                            # 这里简单处理，实际应用中可能需要更复杂的逻辑
                            log.error(f"Lane status shape mismatch: {lane_status.shape} vs {lane_status_shape}")
                            if np.prod(lane_status.shape) <= np.prod(lane_status_shape):
                                # 如果实际数据小于缓冲区，创建一个新的全零数组，然后复制数据
                                lane_status_array.fill(0)
                                # 获取可复制的最小维度
                                min_dims = [min(lane_status.shape[i], lane_status_shape[i]) for i in
                                            range(min(len(lane_status.shape), len(lane_status_shape)))]
                                # 复制数据
                                if len(min_dims) == 3:
                                    lane_status_array[:min_dims[0], :min_dims[1], :min_dims[2]] = lane_status[
                                                                                                  :min_dims[0],
                                                                                                  :min_dims[1],
                                                                                                  :min_dims[2]]
                                elif len(min_dims) == 2:
                                    lane_status_array[:min_dims[0], :min_dims[1], 0] = lane_status[:min_dims[0],
                                                                                       :min_dims[1]]
                                elif len(min_dims) == 1:
                                    lane_status_array[:min_dims[0], 0, 0] = lane_status[:min_dims[0]]
                            else:
                                # 如果数据太大，截断
                                slices = tuple(slice(0, dim) for dim in lane_status_shape)
                                lane_status_array[:] = lane_status[slices]
                        else:
                            lane_status_array[:] = lane_status

                        shared_data["lane_status"] = True  # 标记lane_status已更新
                    else:
                        # 如果lane_status不是numpy数组，则直接存储
                        try:
                            shared_data["lane_status"] = pickle.dumps(status["lane_status"])
                        except Exception as e:
                            log.error(f"Error serializing lane_status: {e}")
                            shared_data["lane_status"] = None
                else:
                    shared_data["lane_status"] = None

                # Update last update time
                last_update_time = current_time
            else:
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)

    except KeyboardInterrupt:
        log.debug("Main process: Terminating child process...")
        running_event.clear()
        p_brain.terminate()
        p_brain.join()
        log.debug("Main process: Exiting.")
    except Exception as e:
        log.error(f"An error occurred in main: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()