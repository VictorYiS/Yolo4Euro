# window.py
import re

import cv2
import numpy as np
import pytesseract
import utils
from utils.change_window import check_window_resolution_same


# 基类，封装静态 offset 和 frame
class BaseWindow:
    offset_x = 0
    offset_y = 0
    frame = None
    all_windows = []
    cached_templates = {}

    def __init__(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.color = None
        BaseWindow.all_windows.append(self)

    @staticmethod
    def set_offset(offset_x, offset_y):
        BaseWindow.offset_x = offset_x
        BaseWindow.offset_y = offset_y

    @staticmethod
    def set_frame(frame):
        BaseWindow.frame = frame

    def extract_region(self):
        if BaseWindow.frame is None:
            print("No frame received.")
            return None
        return BaseWindow.frame[
            self.sy + BaseWindow.offset_y : self.ey + 1 + BaseWindow.offset_y,
            self.sx + BaseWindow.offset_x : self.ex + 1 + BaseWindow.offset_x,
        ]

    def update(self):
        self.color = self.extract_region()

    @staticmethod
    def update_all():
        for window in BaseWindow.all_windows:
            window.update()

    @staticmethod
    def load_template_once(template_image_path):
        """
        加载模板图像并缓存，如果已经加载过则直接使用缓存。
        """
        if template_image_path not in BaseWindow.cached_templates:
            template = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise FileNotFoundError(
                    f"Failed to load template image from {template_image_path}"
                )
            BaseWindow.cached_templates[template_image_path] = template
        return BaseWindow.cached_templates[template_image_path]

    def check_similarity(self, template_image_path, threshold=0.8):
        """
        检查窗口区域内是否包含指定图像，并返回相似度。

        参数:
        - template_image_path: 模板图像的路径
        - threshold: 相似度阈值，默认为0.8

        返回:
        - 如果相似度超过阈值，返回True，否则返回False
        - 匹配的相似度值
        """
        if self.gray is None:
            print("No grayscale data to compare.")
            return False, 0.0

        # 加载模板图像，仅加载一次
        template_image = BaseWindow.load_template_once(template_image_path)

        # 进行模板匹配
        result = cv2.matchTemplate(self.gray, template_image, cv2.TM_CCOEFF_NORMED)

        # 查找匹配结果
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # 返回匹配结果
        return max_val >= threshold, max_val

    def __repr__(self):
        return f"BaseWindow(sx={self.sx}, sy={self.sy}, ex={self.ex}, ey={self.ey}, offset_x={BaseWindow.offset_x}, offset_y={BaseWindow.offset_y})"


# 新的基类，统一返回状态（0/1 或 百分比）
class StatusWindow(BaseWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.status = 0  # 初始化为0

    def update(self):
        super().update()
        if self.color is not None:
            self.process_color()
        else:
            self.status = 0

    def process_color(self):
        # 子类需要实现具体的处理逻辑
        pass

    def get_status(self):
        return self.status


# 数值窗口类，用于扫描窗口中的数值大小
class NumberWindow(StatusWindow):
    def __init__(self, sx, sy, ex, ey, min_value=0, max_value=100):
        super().__init__(sx, sy, ex, ey)
        self.value = 0
        self.min_value = min_value
        self.max_value = max_value
        self.gray = None

    def process_color(self):
        # 转换为灰度图
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        # 对图像进行预处理以提高OCR精度
        _, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 使用pytesseract进行OCR识别
        try:
            # 配置pytesseract只识别数字
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
            text = pytesseract.image_to_string(binary, config=custom_config).strip()

            # 尝试提取数值
            if text:
                # 使用正则表达式提取数字
                number_match = re.search(r'\d+(\.\d+)?', text)
                if number_match:
                    self.value = float(number_match.group())
                    # 确保值在设定范围内
                    self.value = max(self.min_value, min(self.max_value, self.value))
                    # 计算百分比状态
                    if self.max_value > self.min_value:
                        self.status = (self.value - self.min_value) / (self.max_value - self.min_value)
                    else:
                        self.status = 0
                else:
                    self.value = 0
                    self.status = 0
            else:
                self.value = 0
                self.status = 0
        except Exception as e:
            print(f"OCR Error: {e}")
            self.value = 0
            self.status = 0

    def get_value(self):
        return self.value

    def __repr__(self):
        return f"NumberWindow(value={self.value}, status={self.status:.2f}, sx={self.sx}, sy={self.sy}, ex={self.ex}, ey={self.ey})"


# 时间窗口类，用于扫描窗口中的hh:mm格式时间
class TimeWindow(NumberWindow):
    def __init__(self, sx, sy, ex, ey):
        # 调用父类构造函数，但不使用min_value和max_value
        super().__init__(sx, sy, ex, ey)
        self.hours = 0
        self.minutes = 0
        self.total_minutes = 0
        # 一天的总分钟数作为最大值用于计算状态
        self.max_minutes = 24 * 60

    def process_color(self):
        # 转换为灰度图
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        # 对图像进行预处理以提高OCR精度
        _, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 使用pytesseract进行OCR识别
        try:
            # 配置pytesseract识别时间格式
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'
            text = pytesseract.image_to_string(binary, config=custom_config).strip()

            # 使用正则表达式匹配hh:mm格式
            time_match = re.search(r'(\d{1,2}):(\d{2})', text)
            if time_match:
                self.hours = int(time_match.group(1))
                self.minutes = int(time_match.group(2))

                # 验证时间有效性
                if 0 <= self.hours <= 23 and 0 <= self.minutes <= 59:
                    # 计算总分钟数
                    self.total_minutes = self.hours * 60 + self.minutes
                    # 计算状态（一天中的百分比）
                    self.status = self.total_minutes / self.max_minutes
                else:
                    self.hours = 0
                    self.minutes = 0
                    self.total_minutes = 0
                    self.status = 0
            else:
                self.hours = 0
                self.minutes = 0
                self.total_minutes = 0
                self.status = 0
        except Exception as e:
            print(f"Time OCR Error: {e}")
            self.hours = 0
            self.minutes = 0
            self.total_minutes = 0
            self.status = 0

    def get_time_string(self):
        """返回hh:mm格式的时间字符串"""
        return f"{self.hours:02d}:{self.minutes:02d}"

    def get_total_minutes(self):
        """返回从0点开始的总分钟数"""
        return self.total_minutes

    def __repr__(self):
        return f"TimeWindow(time={self.get_time_string()}, minutes={self.total_minutes}, status={self.status:.2f}, sx={self.sx}, sy={self.sy}, ex={self.ex}, ey={self.ey})"


# 查找logo位置的函数
def find_game_window_logo(frame, template_path, threshold):
    return (0, 0)
    # 读取模板图像
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Failed to load template image from {template_path}")
        return None

    # 将frame转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 模板匹配
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

    # 查找匹配区域
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 如果匹配结果足够好，则返回logo的左上角位置
    if max_val >= threshold:
        return max_loc
    else:
        return None


# 设置窗口的相对坐标偏移
def set_windows_offset(frame):
    # 查找logo的初始位置
    logo_position = find_game_window_logo(frame, "./images/title_logo.png", 0.8)

    if logo_position is not None:
        offset_x, offset_y = logo_position

        # 根据logo图片再title bar的位置修正
        # offset_x += 10 # 不需要，已经校正窗口位置了
        offset_y += 45

        # 设置偏移量给所有窗口对象
        BaseWindow.set_offset(offset_x, offset_y)
        BaseWindow.set_frame(frame)
        BaseWindow.update_all()

        print(f"All windows offset by ({offset_x}, {offset_y})")
        return True
    else:
        print("Failed to find the game logo, offsets not set.")
        return False


# 实际游戏窗口大小
game_width = 1920  # NOTE: 替换成你游戏的宽度和分辨率
game_height = round(game_width * 0.5625)

base_width = 1920 # 勿动
base_height = 1080 # 勿动

# 计算缩放因子
width_scale = game_width / base_width
height_scale = game_height / base_height

print(f"width_scale: {width_scale}, height_scale: {height_scale}")


# 坐标转换函数
def convert_coordinates(x1, y1, x2, y2):
    # return x1, y1, x2, y2
    new_x1 = round(x1 * width_scale)
    new_y1 = round(y1 * height_scale)
    new_x2 = round(x2 * width_scale)
    new_y2 = round(y2 * height_scale)
    return new_x1, new_y1, new_x2, new_y2


game_window = BaseWindow(0, 0, game_width, game_height)
# 转换后的窗口坐标

# 根据游戏调整
self_speed_window = NumberWindow(*convert_coordinates(1486, 706, 1509, 729))
self_distance_window = NumberWindow(*convert_coordinates(1591, 739, 1623, 755))
self_time_window = TimeWindow(1830, 707, 1884, 726)

roi_x_size = 1000  # ROI的宽度和高度（以游戏窗口中心为中心的矩形）
roi_y_size = 1200  # ROI的宽度和高度（以游戏窗口中心为中心的矩形）
start_xy = (
    game_width // 2 - roi_x_size // 2,
    game_height // 2 - roi_y_size // 2 - 100,
)  # ROI的起始坐标 (x, y)


battle_roi_window = BaseWindow(
    start_xy[0], start_xy[1], start_xy[0] + roi_x_size, start_xy[1] + roi_y_size
)
