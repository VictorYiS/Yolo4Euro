import cv2
import time
from window import *
import grabscreen
import signal
import sys
import hashlib
import colorsys
import tkinter as tk
import utils.change_window as change_window

# 标志位，表示是否继续运行
running = True


# 处理 Ctrl+C 的函数
def signal_handler(sig, frame):
    global running
    print("\nGracefully exiting...")
    running = False


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)


# 等待游戏窗口出现的函数
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

    # 循环检测窗口是否被关闭
    while True:
        # 监听窗口关闭事件
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

        # 存储变量及其对应的标签
        self.variables = {}

    def add_variable(self, var_name, var_type="float"):
        """
        添加一个新的追踪变量到GUI。

        :param var_name: 变量的名称，用于显示和更新
        :param var_type: 变量的类型，'float' 或 'bool'
        :param column: 'left' 或 'right'，决定标签显示在哪一栏
        """
        frame = self.frame

        # 创建标签
        label = tk.Label(frame, text=f"{var_name}: 0.00")
        label.pack(anchor="w", pady=2)

        # 存储变量信息
        self.variables[var_name] = {"type": var_type, "label": label}

    def update_status(self, **kwargs):
        """
        更新多个变量的状态。

        :param kwargs: 以变量名为键，变量值为值的键值对
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


# 主程序循环，显示玩家的血条数值，并支持优雅退出
def main_loop():
    root = tk.Tk()
    app = GameStatusApp(root)

    # 添加初始变量（示例）

    app.add_variable("self_speed", var_type="integer")
    app.add_variable("self_distance", var_type="integer")

    app.add_variable("self_time", var_type="String")

    if wait_for_game_window():
        display_gui_elements()

        # 进入主循环
        while running:
            frame = grabscreen.grab_screen()
            BaseWindow.set_frame(frame)
            BaseWindow.update_all()

            # 更新 Tkinter 界面上的状态
            app.update_status(
                **{
                    "self_speed": self_speed_window.get_status(),
                    "self_distance": self_distance_window.get_status(),
                    "self_time": self_time_window.get_status()
                }
            )

            # 更新 Tkinter 窗口
            root.update_idletasks()
            root.update()


if __name__ == "__main__":
    time.sleep(1.0)
    grabscreen.init_camera(target_fps=30)

    change_window.correction_window()

    if check_window_resolution_same(game_width, game_height) == False:
        raise ValueError(
            f"游戏分辨率和配置game_width({game_width}), game_height({game_height})不一致，请到window.py中修改"
        )

    print("start main_loop")
    main_loop()
    print("Program has exited cleanly.")
