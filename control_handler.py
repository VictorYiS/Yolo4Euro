# Keyboard controller setup

import multiprocessing as mp
import queue
import threading
import time
import traceback

import cv2
from pynput import keyboard

from driver import TruckController

from log import log


class ActionThread(threading.Thread):
    def __init__(self, normal_queue, kb_controller):
        threading.Thread.__init__(self, daemon=True)
        self.normal_queue = normal_queue
        self.kb_controller = kb_controller
        self.running = True

    def run(self):
        """Process actions from queues with priority for emergency actions"""
        log.info("Action thread started")
        while self.running:
                # If no emergency actions, check normal queue
                try:
                    # Wait a bit for normal actions
                    action = self.normal_queue.get(timeout=0.05)
                    log.debug(f"Executing normal action: {action}")
                    self._execute_action(action)
                    self.normal_queue.task_done()
                except queue.Empty:
                    # No actions to process
                    time.sleep(0.01)
                except Exception as e:
                    log.error(f"Error in action thread: {e}")
                    time.sleep(0.1)  # Prevent high CPU usage on error

    def _execute_action(self, action):
        # Expected action format is a dict with at least 'key' and 'duration'
        if 'key' in action and 'duration' in action:
            self.kb_controller.press_key(action['key'], action['duration'])
        else:
            log.warning(f"Invalid action format: {action}")

    def stop(self):
        """Stop the action thread"""
        self.running = False
        log.info("Stopping action thread")
        self.join()


class KeyController:
    def __init__(self, truck_controller, running_mode):
        self.controller = keyboard.Controller()
        self.key_press_interval = 0.05  # Reduced from 0.1 for more responsive controls
        self.last_press_time = {}  # Track last press time for each key

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
            elif key.char == 'x':
                # Emergency stop
                if self.truck_controller.get_drive_status():
                    self.truck_controller.drive_mode_toggle()
                    log.info("Emergency stop activated")
                    self.press_key('s', 1.0)  # Apply brakes
            elif key == keyboard.Key.esc:
                # Exit program
                self.running_mode.clear()
                log.info("ESC pressed, exiting program")
                return False  # Stop listener
        except AttributeError:
            # Special key handling
            log.error(f"Special key pressed error: {key}, {traceback.format_exc()}")

        return True  # Continue listening

    def press_key(self, key, duration=0.1):
        """Press and hold a key for a specific duration"""
        current_time = time.time()

        # Rate limiting to prevent excessive key presses
        if key in self.last_press_time and current_time - self.last_press_time.get(key, 0) < self.key_press_interval:
            return

        try:
            self.controller.press(key)
            time.sleep(duration)
            self.controller.release(key)
            self.last_press_time[key] = current_time
            log.debug(f"Pressed key {key} for {duration}s")
        except Exception as e:
            log.error(f"Error pressing key {key}: {e}")


class ProcessingManager:
    def __init__(self, shared_data, running_event):
        self.shared_data = shared_data
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
        """获取当前状态"""
        # 从共享数据结构中读取状态
        return dict(self.shared_data)

    def clear_event_queues(self):
        """清空事件队列"""
        while not self.normal_queue.empty():
            self.normal_queue.get_nowait()

    def handle_action_execution(self, action):
        """处理动作执行和事件监控，需要继续改造"""
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

    def _normal_running(self):
        if self.truck_controller.get_drive_status():
            status = self.get_current_state()
            action = None
            if status.get('detect_frame') is not None:
                # 处理图像和状态
                action = self.truck_controller.get_action(status)
            if action:
                self.handle_action_execution(action)

    def _handle_normal_events(self, action):
        """处理普通事件"""
        self.normal_queue.put(action)


def process(shared_data, running_event):
    """主入口函数"""
    manager = ProcessingManager(shared_data, running_event)
    manager.run()