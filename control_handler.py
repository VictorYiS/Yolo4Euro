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


class ActionThread(threading.Thread):
    def __init__(self, action_queue, kb_controller):
        threading.Thread.__init__(self, daemon=True)
        self.action_queue = action_queue
        self.kb_controller = kb_controller
        self.running = True

        # Key state tracking
        self.key_states = {
            'direction': {'key': None, 'until': 0},
            'movement': {'key': None, 'until': 0}
        }

        # Timing parameters
        self.min_w_duration = 0.08  # Minimum forward duration for smoother movement
        self.key_check_interval = 0.01  # More responsive checks

        # Performance tracking
        self.last_action_time = time.time()
        self.action_count = 0

    def run(self):
        """Process actions with improved timing and state management"""
        log.info("Action thread started")
        last_check_time = time.time()

        while self.running:
            try:
                current_time = time.time()

                # Process any new actions
                self._process_new_actions(current_time)

                # Update key states more frequently
                if current_time - last_check_time >= self.key_check_interval:
                    self._update_key_states(current_time)
                    last_check_time = current_time

                # Prevent CPU overload
                time.sleep(0.001)
            except Exception as e:
                log.error(f"Action thread error: {e}")
                traceback.print_exc()
                time.sleep(0.01)  # Reduce CPU usage on error

    def _process_new_actions(self, current_time):
        """Process all available actions with intelligent batching"""
        # Collect actions for batching
        batch = []
        try:
            # Collect up to 5 actions or until queue is empty
            while len(batch) < 5:
                action = self.action_queue.get(block=False)
                batch.append(action)
        except queue.Empty:
            pass

        if not batch:
            return

        # Process batched actions
        self._apply_batched_actions(batch, current_time)

        # Update metrics
        self.action_count += len(batch)
        if current_time - self.last_action_time > 10.0:
            log.debug(f"Action rate: {self.action_count / 10.0:.1f} actions/sec")
            self.action_count = 0
            self.last_action_time = current_time

    def _apply_batched_actions(self, actions, current_time):
        """Apply multiple similar actions efficiently as a single action with longer duration"""
        # Separate by action type
        direction_actions = {}
        movement_actions = {}

        # Categorize and combine similar actions
        for action in actions:
            if 'direction' in action and action['direction'] != 'none':
                key = action['direction']
                direction_actions[key] = direction_actions.get(key, 0) + float(action.get('duration', 0.1))

            if 'movement' in action and action['movement'] != 'none':
                key = action['movement']
                movement_actions[key] = movement_actions.get(key, 0) + float(action.get('duration', 0.1))

        # Apply the dominant direction action (if any)
        if direction_actions:
            best_dir = max(direction_actions.items(), key=lambda x: x[1])
            duration = min(best_dir[1], 0.3)  # Cap duration to prevent overly long presses
            self._set_key_state('direction', best_dir[0], current_time + duration)

        # Apply the dominant movement action (prioritize 'w' for continuous acceleration)
        if 'w' in movement_actions:
            # Ensure 'w' is held longer for smoother movement
            duration = max(movement_actions['w'], self.min_w_duration)
            duration = min(duration, 0.25)  # Cap maximum duration
            self._set_key_state('movement', 'w', current_time + duration)
        elif movement_actions:
            # Apply other movement keys (s, etc.)
            best_mov = max(movement_actions.items(), key=lambda x: x[1])
            duration = min(best_mov[1], 0.2)  # Cap duration
            self._set_key_state('movement', best_mov[0], current_time + duration)

    def _set_key_state(self, state_type, key, until_time):
        """Set a key state with improved conflict handling"""
        current = self.key_states[state_type]['key']

        # Handle key conflicts
        if current != key:
            # Release current key if different
            if current is not None:
                self.kb_controller.release_key(current)

            # Press new key
            if key != 'none':
                self.kb_controller.press_key_without_release(key)

            # Update state
            self.key_states[state_type]['key'] = key if key != 'none' else None

        # Update expiration time (extend if same key)
        if key != 'none':
            self.key_states[state_type]['until'] = until_time

    def _update_key_states(self, current_time):
        """Release expired keys"""
        for state_type, state in self.key_states.items():
            if state['key'] and current_time > state['until']:
                self.kb_controller.release_key(state['key'])
                state['key'] = None

    def clear_queue(self):
        """Clear action queue"""
        try:
            while True:
                self.action_queue.get_nowait()
        except queue.Empty:
            pass

    def stop(self):
        """Stop action thread and release all keys"""
        self.running = False

        # Release all active keys
        for state_type, state in self.key_states.items():
            if state['key']:
                self.kb_controller.release_key(state['key'])
                state['key'] = None

        log.info("Stopping action thread")


class KeyController:
    def __init__(self, truck_controller, running_mode):
        self.controller = keyboard.Controller()
        self.key_press_lock = threading.Lock()  # Prevent concurrent key operations
        self.current_pressed_keys = set()  # Track currently pressed keys

        # Set up keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

        # Controllers
        self.truck_controller = truck_controller
        self.running_mode = running_mode

    def on_key_press(self, key):
        """Handle global keyboard controls"""
        try:
            if hasattr(key, 'char'):
                if key.char == 'z':
                    # Toggle autodrive mode
                    self.truck_controller.drive_mode_toggle()
                    log.info(f"AutoDrive {'activated' if self.truck_controller.get_drive_status() else 'deactivated'}")
                    self.running_mode.set()
                elif key.char == 'x':
                    # Emergency stop
                    if self.truck_controller.get_drive_status():
                        self.truck_controller.drive_mode_toggle()
                        log.info("Emergency stop activated")
                        self.press_key('s', 0.1)  # Brief brake press
                        self.running_mode.clear()
            elif key == keyboard.Key.esc:
                # Exit program
                self.running_mode.clear()
                log.info("ESC pressed, exiting program")
                return False  # Stop listener
        except AttributeError:
            # Special key handling
            pass

        return True  # Continue listening

    def press_keys(self, direction_key, movement_key, duration=0.1):
        """Press and hold two keys for a specified duration"""
        with self.key_press_lock:
            try:
                # Handle conflicting keys
                self._handle_key_conflicts(direction_key, movement_key)

                # Press keys
                if direction_key != 'none':
                    self.controller.press(direction_key)
                    self.current_pressed_keys.add(direction_key)

                if movement_key != 'none':
                    self.controller.press(movement_key)
                    self.current_pressed_keys.add(movement_key)

                # Wait for duration
                time.sleep(duration)

                # Release keys
                if direction_key != 'none':
                    self.controller.release(direction_key)
                    self.current_pressed_keys.discard(direction_key)

                if movement_key != 'none':
                    self.controller.release(movement_key)
                    self.current_pressed_keys.discard(movement_key)

                log.debug(f"Key operation: {direction_key}/{movement_key} for {duration:.2f}s")
            except Exception as e:
                log.error(f"Key error {direction_key}/{movement_key}: {e}")
                # Ensure keys are released on error
                self._release_all_keys()

    def press_key_without_release(self, key):
        """Press a key without releasing (for continuous press)"""
        if key == 'none':
            return

        with self.key_press_lock:
            try:
                # Handle conflicting keys
                self._handle_single_key_conflict(key)

                # Press key
                self.controller.press(key)
                self.current_pressed_keys.add(key)
                # log.debug(f"Key down: {key}")
            except Exception as e:
                log.error(f"Key press error {key}: {e}")

    def release_key(self, key):
        """Release a specific key"""
        if key == 'none':
            return

        with self.key_press_lock:
            if key in self.current_pressed_keys:
                self.controller.release(key)
                self.current_pressed_keys.discard(key)
                # log.debug(f"Key up: {key}")

    def _handle_key_conflicts(self, direction_key, movement_key):
        """Handle conflicts between opposing keys"""
        # Handle direction key conflicts
        if direction_key != 'none':
            self._handle_single_key_conflict(direction_key)

        # Handle movement key conflicts
        if movement_key != 'none':
            self._handle_single_key_conflict(movement_key)

    def _handle_single_key_conflict(self, key):
        """Handle conflict for a single key"""
        # Define opposing key pairs
        opposing_pairs = {
            'a': 'd', 'd': 'a',  # Left/right
            'w': 's', 's': 'w'  # Forward/backward
        }

        # Release opposing key if pressed
        if key in opposing_pairs and opposing_pairs[key] in self.current_pressed_keys:
            self.controller.release(opposing_pairs[key])
            self.current_pressed_keys.discard(opposing_pairs[key])

    def _release_all_keys(self):
        """Release all currently pressed keys"""
        with self.key_press_lock:
            for key in list(self.current_pressed_keys):
                self.controller.release(key)
            self.current_pressed_keys.clear()
            log.debug("Released all keys")

    def press_key(self, key, duration=0.1):
        """Press a single key for a specified duration"""
        self.press_keys(key, 'none', duration)


class ProcessingManager:
    def __init__(self, shared_data, lane_status_buffer, lane_status_shape, running_event):
        self.shared_data = shared_data
        self.lane_status_buffer = lane_status_buffer
        self.lane_status_shape = lane_status_shape
        self.running_event = running_event

        # Initialize queues
        self.action_queue = mp.Queue()

        # Initialize controllers
        self.truck_controller = TruckController()
        self.running_mode = threading.Event()
        self.running_mode.clear()
        self.kb_controller = KeyController(self.truck_controller, self.running_mode)

        # Initialize action thread
        self.action_executor = ActionThread(self.action_queue, self.kb_controller)

        # Performance monitoring
        self.last_state_update = time.time()
        self.frame_times = []
        self.max_frame_history = 60  # ~1 second at 60fps

    def get_current_state(self):
        """Get current state with lane data from shared memory"""
        # Start timing frame processing
        frame_start = time.time()

        # Read basic state from shared data
        state = dict(self.shared_data)

        # Process serialized data
        self._deserialize_state_data(state)

        # Track frame processing time
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)

        # Log performance metrics periodically
        current_time = time.time()
        if current_time - self.last_state_update > 5.0:
            self._log_performance_metrics()
            self.last_state_update = current_time

        return state

    def _deserialize_state_data(self, state):
        """Safely deserialize data in the state dictionary"""
        # Deserialize car detection and frame data
        # state["car_detect"] = self._get_serialized_data(state, "car_detect")
        state["traffic_detect"] = self._get_serialized_data(state, "traffic_detect")
        state["detect_frame"] = self._get_serialized_data(state, "detect_frame")

        # Handle lane_status data
        if state.get("lane_status") is True:
            # Restore numpy array from shared memory
            try:
                lane_array = np.frombuffer(self.lane_status_buffer, dtype=np.float32).reshape(
                    self.lane_status_shape)
                state["lane_status"] = lane_array.copy()  # Copy to avoid data race
            except Exception as e:
                log.error(f"Error retrieving lane_status from shared memory: {e}")
                state["lane_status"] = None
        elif state.get("lane_status") is not None and not isinstance(state["lane_status"], bool):
            # Deserialize serialized object
            try:
                state["lane_status"] = pickle.loads(state["lane_status"])
            except Exception as e:
                log.error(f"Error deserializing lane_status: {e}")
                state["lane_status"] = None

    def _get_serialized_data(self, state, key):
        """Safely deserialize data from the state dictionary"""
        if key in state and state[key]:
            try:
                return pickle.loads(state[key])
            except Exception as e:
                log.error(f"Error deserializing {key}: {e}")
                return None
        return state.get(key)

    def _log_performance_metrics(self):
        """Log performance metrics"""
        if not self.frame_times:
            return

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1 / avg_frame_time if avg_frame_time > 0 else 0

        log.info(f"Avg processing time: {avg_frame_time:.4f}s, "
                 f"FPS: {fps:.1f}")

    def clear_action_queue(self):
        """Clear action queue to prevent backlog"""
        self.action_executor.clear_queue()

    def handle_action(self, action):
        """Process an action by adding it to the queue"""
        for cmd in action:
            if isinstance(cmd, list) and len(cmd) == 2:
                # Parse direction and movement
                direction_parts = cmd[0].split(':')
                movement_parts = cmd[1].split(':')

                if len(direction_parts) == 2 and len(movement_parts) == 2:
                    direction_key = direction_parts[0]
                    direction_duration = float(direction_parts[1])

                    movement_key = movement_parts[0]
                    movement_duration = float(movement_parts[1])

                    # Create action object
                    action_obj = {
                        'direction': direction_key,
                        'movement': movement_key,
                        'duration': max(direction_duration, movement_duration)
                    }

                    # Add to queue
                    self.action_queue.put(action_obj)
                else:
                    log.warning(f"Invalid command format: {cmd}")

    def run(self):
        """Main processing loop with improved error handling"""
        try:
            # Start action thread
            self.action_executor.start()
            log.info("Processing manager started")

            # Main loop
            while self.running_event.is_set():
                if self.running_mode.is_set():
                    self._process_frame()
                else:
                    # Idle mode
                    self.clear_action_queue()
                    time.sleep(0.03)
        except KeyboardInterrupt:
            log.info("Exiting due to keyboard interrupt...")
            self.running_event.clear()
        except Exception as e:
            error_message = traceback.format_exc()
            log.error(f"Error occurred: {e}\n{error_message}")
            self.running_event.clear()
        finally:
            # Clean up
            self.action_executor.stop()
            cv2.destroyAllWindows()
            log.info("Processing manager stopped")

    def _process_frame(self):
        """Process current frame and execute actions"""
        if not self.truck_controller.get_drive_status():
            # Autodrive not active, skip processing
            return

        # Get current state
        status = self.get_current_state()
        speed = status.get("speed", 0)
        traffic_detect = status.get("traffic_detect", None)

        try:
            # Get action from truck controller
            action = self.truck_controller.get_action(status)
            # initial_action = action.copy()

            # Ensure there's always some forward momentum at low speeds
            if not action and speed < 20 and (traffic_detect is None):
                # Add light acceleration to maintain movement
                action = [['none:0.08', 'w:0.08']]

            # with open("debug_action.txt", "a") as f:
            #     frame_time = status.get("detect_frame", None)
            #     if action != initial_action:
            #         f.write(f"Action changed from {initial_action} to {action}\n in status: {frame_time}\n")
            #     else:
            #         f.write(f"{action} in status: {frame_time}\n")

            # Execute action
            if action:
                self.handle_action(action)

        except Exception as e:
            log.error(f"Error getting action: {e}")
            traceback.print_exc()

            # Fallback to simple forward action on error
            fallback_action = [['none:0.03', 'w:0.03']]
            self.handle_action(fallback_action)


def process(shared_data, lane_status_buffer, lane_status_shape, running_event):
    """Main process entry function with improved error handling"""
    try:
        log.info("Starting processing manager")
        manager = ProcessingManager(shared_data, lane_status_buffer, lane_status_shape, running_event)
        manager.run()
    except Exception as e:
        log.error(f"Fatal error in process: {e}")
        traceback.print_exc()
        running_event.clear()