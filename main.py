import multiprocessing as mp
import time
import sys
import signal
import cv2

from dashboard import TruckDashboard
from utils import change_window
import window
import grabscreen
from log import log
from control_handler import process

# from utils.change_window import *

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
        "objects": [],
        "detect_frame": None
    })

    # Create and initialize Dashboard
    dashboard = TruckDashboard()

    # Start child process
    p_brain = mp.Process(target=process, args=(shared_data, running_event))
    p_brain.start()

    try:
        while running_event.is_set():
            dashboard.update_data()
            # 更新共享数据
            status = dashboard.get_stored_data()
            # 不能直接传递OpenCV的图像对象，需要标记更新
            status_copy = status.copy()
            status_copy["detect_frame"] = True if status["detect_frame"] is not None else False
            shared_data.update(status_copy)
    except KeyboardInterrupt:
        log.debug("Main process: Terminating child process...")
        running_event.clear()
        p_brain.terminate()
        p_brain.join()
        log.debug("Main process: Exiting.")
    except Exception as e:
        log.debug(f"An error occurred in main: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()