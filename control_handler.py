# Keyboard controller setup

import multiprocessing as mp
import queue
import threading
import time
import traceback
import pickle
import numpy as np

import cv2
from pynput import keyboard

from driver import TruckController

from log import log


# Add to the top of the file in control_handler.py or create a new version

### modification: Improve action execution in ActionThread class
class ActionThread(threading.Thread):
    def __init__(self, normal_queue, kb_controller):
        threading.Thread.__init__(self, daemon=True)
        self.normal_queue = normal_queue
        self.kb_controller = kb_controller
        self.running = True
        self.last_action = None
        self.last_action_time = time.time()
        self.min_action_interval = 0.01  # Minimum time between actions

    def run(self):
        """Process actions from queues with priority for emergency actions"""
        log.info("Action thread started")
        while self.running:
            try:
                # Wait a bit for normal actions
                action = self.normal_queue.get(timeout=0.02)  # Reduced timeout for faster response

                # Check if we need to throttle rapid key presses
                current_time = time.time()
                time_since_last = current_time - self.last_action_time

                if time_since_last < self.min_action_interval:
                    # Too soon for another action, slight sleep to prevent CPU hogging
                    time.sleep(max(0.001, self.min_action_interval - time_since_last))

                log.debug(f"Executing normal action: {action}")
                self._execute_action(action)
                self.last_action = action
                self.last_action_time = time.time()

            except queue.Empty:
                # No actions to process
                time.sleep(0.001)  # Very short sleep to reduce CPU usage
            except Exception as e:
                log.error(f"Error in action thread: {e}")
                time.sleep(0.05)  # Prevent high CPU usage on error

    def _execute_action(self, action):
        # Execute the action with the specified duration
        if 'key' in action and 'duration' in action:
            # Ensure minimum duration for key presses to be registered by the game
            duration = max(0.01, float(action['duration']))
            self.kb_controller.press_key(action['key'], duration)
        else:
            log.error(f"Invalid action format: {action}")

    def stop(self):
        """Stop the action thread"""
        self.running = False
        log.info("Stopping action thread")
        self.join()


### modification: Improve KeyController for more responsive control
class KeyController:
    def __init__(self, truck_controller, running_mode):
        self.controller = keyboard.Controller()
        self.key_press_interval = 0.03  # Further reduced from 0.05 for more responsive controls
        self.last_press_time = {}  # Track last press time for each key
        self.current_pressed_keys = set()  # Track currently pressed keys

        # Set up keyboard listener for toggle commands
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()
        self.truck_controller = truck_controller
        self.running_mode = running_mode

    def on_key_press(self, key):
        try:
            if key.char == 'z':
                # Toggle autodrive
                self.truck_controller.drive_mode_toggle()
                log.info(f"AutoDrive {'activated' if self.truck_controller.get_drive_status() else 'deactivated'}")
                self.running_mode.set()  # 打开自动驾驶时设置运行模式
            elif key.char == 'x':
                # Emergency stop
                if self.truck_controller.get_drive_status():
                    self.truck_controller.drive_mode_toggle()
                    log.info("Emergency stop activated")
                    self.press_key('s', 0.1)  # Shorter brake time but still effective
                    self.running_mode.clear()  # 停止自动驾驶时清除运行模式
            elif key == keyboard.Key.esc:
                # Exit program
                self.running_mode.clear()
                log.info("ESC pressed, exiting program")
                return False  # Stop listener
        except AttributeError:
            # Special key handling
            pass

        return True  # Continue listening

    def press_key(self, key, duration=0.1):
        """Press and hold a key for a specific duration with improved responsiveness"""
        current_time = time.time()

        # Rate limiting to prevent excessive key presses
        if key in self.last_press_time and current_time - self.last_press_time.get(key, 0) < self.key_press_interval:
            return

        try:
            # Release any conflicting keys first (e.g., don't press W and S simultaneously)
            if key == 'w' and 's' in self.current_pressed_keys:
                self.controller.release('s')
                self.current_pressed_keys.remove('s')
            elif key == 's' and 'w' in self.current_pressed_keys:
                self.controller.release('w')
                self.current_pressed_keys.remove('w')
            elif key == 'a' and 'd' in self.current_pressed_keys:
                self.controller.release('d')
                self.current_pressed_keys.remove('d')
            elif key == 'd' and 'a' in self.current_pressed_keys:
                self.controller.release('a')
                self.current_pressed_keys.remove('a')

            # Press the key
            self.controller.press(key)
            self.current_pressed_keys.add(key)

            # Use shorter sleep for better responsiveness
            time.sleep(duration)

            # Release the key
            self.controller.release(key)
            if key in self.current_pressed_keys:
                self.current_pressed_keys.remove(key)

            self.last_press_time[key] = current_time
            log.debug(f"Pressed key {key} for {duration}s")
        except Exception as e:
            log.error(f"Error pressing key {key}: {e}")


class ProcessingManager:
    def __init__(self, shared_data, lane_status_buffer, lane_status_shape, running_event):
        self.shared_data = shared_data
        self.lane_status_buffer = lane_status_buffer
        self.lane_status_shape = lane_status_shape
        self.running_event = running_event

        # 初始化队列
        self.normal_queue = mp.Queue()

        # 加载配置

        # 初始化智能体
        self.truck_controller = self._initialize_controller()

        # 训练控制
        self.running_mode = threading.Event()
        self.running_mode.clear()

        # 设置键盘监听
        self.kb_controller = KeyController(self.truck_controller, self.running_mode)

        self.action_executor = ActionThread(
            self.normal_queue,
            self.kb_controller
        )

    def _initialize_controller(self):
        return TruckController()

    def get_current_state(self):
        """获取当前状态，包括从共享内存中恢复lane_status"""
        # 从共享数据结构中读取基本状态
        state = dict(self.shared_data)

        state["car_detect"] = self.get_serialized_data(state, "car_detect")
        state["detect_frame"] = self.get_serialized_data(state, "detect_frame")

        # 处理lane_status数据
        if state.get("lane_status") is True:
            # 如果lane_status标记为True，则从共享内存中恢复numpy数组
            try:
                lane_status_array = np.frombuffer(self.lane_status_buffer, dtype=np.float32).reshape(
                    self.lane_status_shape)
                state["lane_status"] = lane_status_array.copy()  # 复制一份，避免数据竞争
            except Exception as e:
                log.error(f"Error retrieving lane_status from shared memory: {e}")
                state["lane_status"] = None
        elif state.get("lane_status") is not None and not isinstance(state["lane_status"], bool):
            # 如果lane_status是序列化的对象，则反序列化
            try:
                state["lane_status"] = pickle.loads(state["lane_status"])
            except Exception as e:
                log.error(f"Error deserializing lane_status: {e}")
                state["lane_status"] = None

        return state

    def get_serialized_data(self, state, key):
        if key in state and state[key]:
            try:
                state[key] = pickle.loads(state[key])
            except Exception as e:
                log.error(f"Error deserializing classes: {e}")
                state[key] = None

        return state[key]


    def clear_event_queues(self):
        """清空事件队列"""
        while not self.normal_queue.empty():
            self.normal_queue.get_nowait()

    def handle_action_execution(self, action):
        """处理动作执行和事件监控"""
        self._handle_normal_events(action)

    def run(self):
        """主运行循环"""
        try:
            self.action_executor.start()
            while self.running_event.is_set():
                if self.running_mode.is_set():
                    self._normal_running()
                else:
                    self.clear_event_queues()
                    time.sleep(0.03)
        except KeyboardInterrupt:
            log.error("进程: 正在退出...")
            self.running_event.clear()
        except Exception as e:
            error_message = traceback.format_exc()
            log.error(f"发生错误: {e}\n{error_message}")
            self.running_event.clear()
        finally:
            self.action_executor.stop()
            cv2.destroyAllWindows()

    ### modification: Improve _normal_running method to reduce control delays
    def _normal_running(self):
        if self.truck_controller.get_drive_status():
            status = self.get_current_state()
            action = None

            try:
                # Process state and get action
                action = self.truck_controller.get_action(status)
            except Exception as e:
                log.error(f"Error getting action: {e}")
                # Fallback to simple forward motion in case of error
                action = [['w', '0.01']]

            if action:
                self.handle_action_execution(action)

    def _handle_normal_events(self, actions):
        """处理普通事件"""
        for cmd in actions:
            if isinstance(cmd, list):
                # 如果是列表，假设第一个元素是键，第二个元素是持续时间
                key = cmd[0]
                duration = float(cmd[1]) if len(cmd) > 1 else 0.05
                action = {'key': key, 'duration': duration}
            else:
                # 否则直接使用命令
                action = {'key': cmd, 'duration': 0.05}
            self.normal_queue.put(action)


def process(shared_data, lane_status_buffer, lane_status_shape, running_event):
    """主入口函数"""
    try:
        manager = ProcessingManager(shared_data, lane_status_buffer, lane_status_shape, running_event)
        manager.run()
    except Exception as e:
        log.error(f"Error in process: {e}")
        traceback.print_exc()