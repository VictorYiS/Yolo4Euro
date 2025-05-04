import keyboard
import time
import pygetwindow as gw


class AutoDriveETS2:
    def __init__(self):
        self.running = False
        self.truck_speed = 1.0  # Adjust based on game speed

    def ensure_game_focus(self):
        try:
            ets2_window = gw.getWindowsWithTitle("Euro Truck Simulator 2")[0]
            ets2_window.activate()
            time.sleep(0.5)
            return True
        except IndexError:
            print("ETS2 not found!")
            return False

    def move_forward(self, duration=1.0):
        keyboard.press('w')
        time.sleep(duration)
        keyboard.release('w')

    def turn_left(self, duration=0.5):
        keyboard.press('a')
        time.sleep(duration)
        keyboard.release('a')

    def turn_right(self, duration=0.5):
        keyboard.press('d')
        time.sleep(duration)
        keyboard.release('d')

    def stop(self):
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')

    def emergency_stop(self):
        self.stop()
        keyboard.press_and_release('esc')

    def run_autodrive(self):
        if not self.ensure_game_focus():
            return

        print("AutoDrive started. Press 'esc' to stop.")
        self.running = True

        try:
            while self.running:
                # Basic driving pattern
                self.move_forward(2)

                # Check for turn signals or road conditions here
                # For now, just simple pattern
                if keyboard.is_pressed('left'):
                    self.turn_left(0.5)
                elif keyboard.is_pressed('right'):
                    self.turn_right(0.5)

                # Emergency stop
                if keyboard.is_pressed('esc'):
                    self.emergency_stop()
                    break

                time.sleep(0.1)  # Prevent CPU overload

        except Exception as e:
            print(f"Error: {e}")
            self.stop()


# Usage
if __name__ == "__main__":
    auto_drive = AutoDriveETS2()
    auto_drive.run_autodrive()