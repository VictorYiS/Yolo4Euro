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
        # Initialize the system components
        initialize_system()

        # Create shared data for inter-process communication
        shared_data, lane_status_buffer, lane_status_shape = setup_shared_data()

        # Create the dashboard instance
        dashboard = TruckDashboard()

        # Start the brain process for control logic
        brain_process = start_brain_process(shared_data, lane_status_buffer, lane_status_shape)

        # Main loop for the dashboard
        run_dashboard_loop(dashboard, shared_data, lane_status_buffer, lane_status_shape)

    except KeyboardInterrupt:
        log.debug("Main process: Captured keyboard interrupt signal...")
        shutdown_system(brain_process)
    except Exception as e:
        log.error(f"An error occurred in the main process: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


def initialize_system():
    """initialize the system components"""
    # set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize the screen system
    grabscreen.init_camera(target_fps=30)

    # Correct the game window
    change_window.correction_window()

    # Check if the game window is set up correctly
    if not change_window.check_window_resolution_same(window.game_width, window.game_height):
        raise ValueError(
            f"The game resolution is inconsistent with the configuration game_width({window.game_width}), "
            f"game_height({window.game_height}), please modify it in window.py"
        )

    # Set the running event to indicate the system is ready
    running_event.set()

    # Wait for the game window to be ready
    if not wait_for_game_window(running_event):
        log.debug("Failed to detect game window.")
        return False

    return True


def setup_shared_data():
    """Set up shared data for inter-process communication"""
    # Create a manager for shared data
    manager = mp.Manager()
    shared_data = manager.dict()

    # Initialize shared data with default values
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

    # Create a shared buffer for lane status
    lane_status_shape = (1200, 600, 3)  # Adjust according to your lane status shape
    buffer_size = int(np.prod(lane_status_shape) * 4)  # 4 bytes for float32
    lane_status_buffer = mp.Array('B', buffer_size)

    return shared_data, lane_status_buffer, lane_status_shape


def start_brain_process(shared_data, lane_status_buffer, lane_status_shape):
    """Start the brain process to handle control logic"""
    brain_process = mp.Process(
        target=process,
        args=(shared_data, lane_status_buffer, lane_status_shape, running_event)
    )
    brain_process.start()
    return brain_process


def run_dashboard_loop(dashboard, shared_data, lane_status_buffer, lane_status_shape):
    """running dashboard loop to update data"""
    update_interval = 0.1
    last_update_time = time.time()

    while running_event.is_set():
        current_time = time.time()

        # only update if enough time has passed
        if current_time - last_update_time >= update_interval:
            dashboard.update_data()

            update_shared_data(dashboard, shared_data, lane_status_buffer, lane_status_shape)

            # update the window with new data
            last_update_time = current_time
        else:
            # sleep to avoid busy waiting
            time.sleep(0.01)


def update_shared_data(dashboard, shared_data, lane_status_buffer, lane_status_shape):
    """update shared data from dashboard"""
    # get the current status from the dashboard
    status = dashboard.get_stored_data()

    update_basic_data(shared_data, status)

    update_lane_status(shared_data, status, lane_status_buffer, lane_status_shape)


def update_basic_data(shared_data, status):
    """update basic data in shared_data"""
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
    """update lane status in shared_data"""
    if status["lane_status"] is None:
        shared_data["lane_status"] = None
        return

    # handle numpy array for lane status
    if isinstance(status["lane_status"], np.ndarray):
        lane_status_array = np.frombuffer(lane_status_buffer, dtype=np.float32).reshape(lane_status_shape)
        lane_status = status["lane_status"]

        if lane_status.shape != lane_status_shape:
            copy_with_shape_handling(lane_status, lane_status_array, lane_status_shape)
        else:
            lane_status_array[:] = lane_status

        shared_data["lane_status"] = True  # mark as updated
    else:
        try:
            shared_data["lane_status"] = pickle.dumps(status["lane_status"])
        except Exception as e:
            log.error(f"error in serialize lane_status data: {e}")
            shared_data["lane_status"] = None


def copy_with_shape_handling(source, target_array, target_shape):
    """handle copying data with shape validation"""
    log.error(f"Lane state shape mismatch: {source.shape} vs {target_shape}")

    if np.prod(source.shape) <= np.prod(target_shape):
        # original data fits into target buffer, fill the target array
        target_array.fill(0)
        min_dims = [min(source.shape[i], target_shape[i]) for i in
                    range(min(len(source.shape), len(target_shape)))]

        # copy data to target array based on the minimum dimensions
        if len(min_dims) == 3:
            target_array[:min_dims[0], :min_dims[1], :min_dims[2]] = source[:min_dims[0], :min_dims[1], :min_dims[2]]
        elif len(min_dims) == 2:
            target_array[:min_dims[0], :min_dims[1], 0] = source[:min_dims[0], :min_dims[1]]
        elif len(min_dims) == 1:
            target_array[:min_dims[0], 0, 0] = source[:min_dims[0]]
    else:
        # source data is larger than target buffer, resize it
        slices = tuple(slice(0, dim) for dim in target_shape)
        target_array[:] = source[slices]


def shutdown_system(brain_process):
    """Clean up and terminate the system"""
    log.debug("Main process: Terminating child process...")
    running_event.clear()
    brain_process.terminate()
    brain_process.join()
    log.debug("Main process: Exiting.")


if __name__ == '__main__':
    main()