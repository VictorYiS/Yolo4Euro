import signal
import time
import tkinter as tk

# from Yolo4Euro.window import self_set_speed

import grabscreen
import utils.change_window as change_window
from window import *

# sign
running = True


# Handler for graceful exit on Ctrl+C
def signal_handler(sig, frame):
    global running
    print("\nGracefully exiting...")
    running = False


# register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


# wait for the game window to be detected and set offsets
def wait_for_game_window():
    while running:
        frame = grabscreen.grab_screen()

        if frame is not None:
            if set_windows_offset(frame):
                print("Game window detected and offsets set!")
                return True

            print("Failed to find the game logo, offsets not set.")

        time.sleep(1)


def display_gui_elements():
    # Ensure that game_window has been updated
    if game_window.color is None:
        print("Game window frame is not available.")
        return

    # Create a copy to draw rectangles on
    game_window_frame = game_window.color.copy()

    # Iterate through all window instances and draw rectangles
    for win in BaseWindow.all_windows:
        # Get the class name of the window instance
        class_name = win.__class__.__name__.replace("Window", "")

        # Define top-left and bottom-right points
        top_left = (win.sx, win.sy)
        bottom_right = (win.ex, win.ey)

        # Draw the rectangle on the game_window_frame
        cv2.rectangle(game_window_frame, top_left, bottom_right, (0, 0, 255), 1)

        text_position = (win.ex + 1, win.sy + 6)
        cv2.putText(
            game_window_frame,
            class_name,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (128, 255, 128),
            1,
            cv2.LINE_AA,
        )

    # Create a window and set it to be always on top
    cv2.namedWindow("Game Window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Game Window", cv2.WND_PROP_TOPMOST, 1)

    # Display the frame with all rectangles
    cv2.imshow("Game Window", game_window_frame)

    # check if the window is still open
    while True:
        # listen for the 'q' key to exit
        if cv2.getWindowProperty("Game Window", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()


class GameStatusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Status")

        self.frame = tk.Frame(root)
        self.frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.variables = {}

    def add_variable(self, var_name, var_type="float"):
        """
        add a variable to the GUI.

        :param var_name: name of the variable to display
        :param var_type: type of the variable, can be 'float', 'bool', 'integer', or 'String'
        """
        frame = self.frame

        # create a label for the variable
        label = tk.Label(frame, text=f"{var_name}: 0.00")
        label.pack(anchor="w", pady=2)

        # set the variable type
        self.variables[var_name] = {"type": var_type, "label": label}

    def update_status(self, **kwargs):
        """
        Update GUI variables with new values.

        :param kwargs: updated values for the variables.
        """
        for var_name, value in kwargs.items():
            if var_name in self.variables:
                var_info = self.variables[var_name]
                var_type = var_info["type"]
                label = var_info["label"]

                if var_type == "float":
                    label.config(text=f"{var_name}: {value:.2f}%")
                elif var_type == "bool":
                    text = "Active" if value else "Inactive"
                    label.config(text=f"{var_name}: {text}")
                else:
                    label.config(text=f"{var_name}: {value}")
            else:
                print(f"Warning: Variable '{var_name}' not found in GUI.")


# Main loop to initialize the GUI and update game status
def main_loop():
    root = tk.Tk()
    app = GameStatusApp(root)

    # Add GUI elements for game status

    app.add_variable("self_speed", var_type="integer")
    app.add_variable("self_distance", var_type="integer")

    app.add_variable("self_time", var_type="String")
    app.add_variable("self_setspeed",var_type="integer")
    app.add_variable("self_gear", var_type="String")

    if wait_for_game_window():
        display_gui_elements()

        # Ensure all windows are initialized
        while running:
            frame = grabscreen.grab_screen()
            BaseWindow.set_frame(frame)
            BaseWindow.update_all()

            # Update the game data
            app.update_status(
                **{
                    "self_speed": self_speed_window.get_status(),
                    "self_distance": self_distance_window.get_status(),
                    "self_time": self_time_window.get_status(),
                    "self_setspeed":self_set_speed.get_status(),
                    "self_gear": self_gear_window.get_status()
                }
            )

            # update the GUI
            root.update_idletasks()
            root.update()


if __name__ == "__main__":
    time.sleep(1.0)
    grabscreen.init_camera(target_fps=30)

    change_window.correction_window()

    if change_window.check_window_resolution_same(game_width, game_height) == False:
        raise ValueError(
            f"The game resolution is inconsistent with the configuration game_width({game_width}), game_height({game_height}), please modify it in window.py"
        )

    print("start main_loop")
    main_loop()
    print("Program has exited cleanly.")
