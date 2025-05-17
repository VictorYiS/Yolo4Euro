# window.py
import os
import re

import cv2
import numpy as np
import pytesseract

from log import log


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


# GearWindow 类，用于识别档位信息 (A1-A12+, N, R1-R3)
class GearWindow(BaseWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.gear_text = "N"  # 默认为空档
        self.gear_type = "neutral"  # 可以是 "advance", "neutral", 或 "reverse"
        self.gear_number = 0  # 档位数字部分
        self.gray = None

    def update(self):
        super().update()
        if self.color is not None:
            self.process_color()

    def get_status(self):
        # 返回档位状态
        return self.gear_type

    def process_color(self):
        if self.color is None:
            log.debug("GearWindow: No color data to process.")

        # 转换为灰度图
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        # 图像预处理优化识别率
        resized = cv2.resize(self.gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            # 使用仅限于可能出现的字符集
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ARN123456789'
            text = pytesseract.image_to_string(binary, config=custom_config).strip()

            # 处理识别的文本
            self._process_gear_text(text)

        except Exception as e:
            print(f"Gear OCR Error: {e}")
            self.reset_gear()

    def _process_gear_text(self, text):
        """处理识别的文本以确定档位状态"""
        # 移除空格并标准化
        text = re.sub(r'\s+', '', text).upper()
        log.debug(f"GearWindow: Processed text: {text}")

        # 匹配档位模式
        advance_match = re.search(r'A(\d+)', text)
        reverse_match = re.search(r'R([123])', text)

        if text == 'N':
            self.gear_text = 'N'
            self.gear_type = 'neutral'
            self.gear_number = 0
        elif advance_match:
            gear_num = int(advance_match.group(1))
            # 验证前进档位范围
            if 1 <= gear_num <= 15:
                self.gear_text = f'A{gear_num}'
                self.gear_type = 'advance'
                self.gear_number = gear_num

        elif reverse_match:
            gear_num = int(reverse_match.group(1))
            # 验证倒退档位范围
            if 1 <= gear_num <= 3:
                self.gear_text = f'R{gear_num}'
                self.gear_type = 'reverse'
                self.gear_number = gear_num

        else:
            # 基于字符存在进行简单匹配
            if 'A' in text:
                nums = re.findall(r'\d+', text)
                if nums and 1 <= int(nums[0]) <= 15:
                    self.gear_text = f'A{nums[0]}'
                    self.gear_type = 'advance'
                    self.gear_number = int(nums[0])
            elif 'R' in text:
                nums = re.findall(r'\d+', text)
                if nums and 1 <= int(nums[0]) <= 3:
                    self.gear_text = f'R{nums[0]}'
                    self.gear_type = 'reverse'
                    self.gear_number = int(nums[0])

    def reset_gear(self):
        """重置档位信息为默认值"""
        self.gear_text = "N"
        self.gear_type = "neutral"
        self.gear_number = 0

    def __repr__(self):
        return f"GearWindow(gear={self.gear_text}, type={self.gear_type}, number={self.gear_number})"


# 数值窗口类，直接返回识别的实际数值
class NumberWindow(StatusWindow):
    def __init__(self, sx, sy, ex, ey, min_value=0, max_value=100):
        super().__init__(sx, sy, ex, ey)
        self.value = 0
        self.min_value = min_value
        self.max_value = max_value
        self.gray = None

    def process_color(self):
        if self.color is None:
            self.value = 0
            self.status = 0  # 保留status属性，但不再用于百分比
            return

        # 转换为灰度图
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        # 增强预处理步骤
        resized = cv2.resize(self.gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        binary = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        try:
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
            text = pytesseract.image_to_string(binary, config=custom_config).strip()

            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if numbers:
                # 如果找到多个数字，选择最长的那个
                longest_num = max(numbers, key=len)
                try:
                    self.value = float(longest_num)
                    # 仍然可以应用范围限制
                    self.value = max(self.min_value, min(self.max_value, self.value))
                    # 直接将识别到的数值赋给status
                    self.status = self.value
                except ValueError:
                    self.value = 0
                    self.status = 0
            else:
                # 如果没有找到数字，保持不变
                log.debug("NumberWindow: No valid number found.")

            # 如果识别结果不可靠，尝试备选方法
            if self.value == 0 and self.min_value > 0:
                self._try_template_matching()

        except Exception as e:
            print(f"OCR Error: {e}")
            self.value = 0
            self.status = 0

    def _try_template_matching(self):
        """尝试使用模板匹配识别数字"""
        # 这里可以实现简单的数字模板匹配逻辑
        pass

    def get_value(self):
        return self.value

    def get_status(self):
        # 重写父类方法，直接返回识别的数值
        return self.value

    def __repr__(self):
        return f"NumberWindow(value={self.value})"



# 时间窗口类，直接返回识别的时间值
class TimeWindow(NumberWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.hours = 0
        self.minutes = 0
        self.total_minutes = 0

    def process_color(self):
        if self.color is None:
            self.reset_time()
            return

        # 转换为灰度图
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        # 时间识别增强预处理
        resized = cv2.resize(self.gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(resized, -1, kernel)
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        try:
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789: '
            text = pytesseract.image_to_string(binary, config=custom_config).strip()

            time_match = re.search(r'(\d{1,2})\s*:\s*(\d{2})', text)

            if not time_match:
                text = re.sub(r'\s+', '', text)  # 移除所有空格
                time_match = re.search(r'(\d{1,2}):(\d{2})', text)

            if time_match:
                self.hours = int(time_match.group(1))
                self.minutes = int(time_match.group(2))

                # 验证时间有效性
                if 0 <= self.hours <= 23 and 0 <= self.minutes <= 59:
                    self.update_time_values()
                else:
                    self.reset_time()
            else:
                # 如果正则匹配失败，尝试分别识别小时和分钟
                self._try_separate_recognition(binary)
        except Exception as e:
            print(f"Time OCR Error: {e}")
            self.reset_time()

    def _try_separate_recognition(self, binary):
        """尝试分别识别小时和分钟部分"""
        height, width = binary.shape

        # 分割图像为左右两部分
        left_half = binary[:, :width // 2]
        right_half = binary[:, width // 2:]

        try:
            left_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
            hours_text = pytesseract.image_to_string(left_half, config=left_config).strip()
            hours_match = re.search(r'(\d{1,2})', hours_text)

            right_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
            minutes_text = pytesseract.image_to_string(right_half, config=right_config).strip()
            minutes_match = re.search(r'(\d{1,2})', minutes_text)

            if hours_match and minutes_match:
                self.hours = int(hours_match.group(1))
                self.minutes = int(minutes_match.group(1))

                # 验证时间有效性
                if 0 <= self.hours <= 23 and 0 <= self.minutes <= 59:
                    self.update_time_values()
                else:
                    self.reset_time()
            else:
                self.reset_time()
        except Exception:
            self.reset_time()

    def reset_time(self):
        """重置时间相关属性"""
        self.hours = 0
        self.minutes = 0
        self.total_minutes = 0
        self.status = 0
        self.value = 0

    def update_time_values(self):
        """更新总分钟数和状态值"""
        self.total_minutes = self.hours * 60 + self.minutes
        # 设置status和value均为总分钟数
        self.status = self.total_minutes
        self.value = self.total_minutes

    def get_time_string(self):
        """返回hh:mm格式的时间字符串"""
        return f"{self.hours:02d}:{self.minutes:02d}"

    def get_total_minutes(self):
        """返回从0点开始的总分钟数"""
        return self.total_minutes

    def get_value(self):
        # 重写父类方法，返回总分钟数
        return self.total_minutes

    def get_status(self):
        # 重写父类方法，返回总分钟数
        return self.total_minutes

    def __repr__(self):
        return f"TimeWindow(time={self.get_time_string()}, minutes={self.total_minutes})"


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
        offset_x += 5 # 不需要，已经校正窗口位置了
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
self_speed_window = NumberWindow(*convert_coordinates(1486, 706, 1513, 729), min_value=0, max_value=90)
self_distance_window = NumberWindow(*convert_coordinates(1585, 739, 1620, 757))
self_time_window = TimeWindow(1831, 707, 1884, 727)
self_set_speed = NumberWindow(*convert_coordinates(1498,783,1519,799))
self_gear_window = GearWindow(1601, 706, 1625, 729)

roi_x_size = 1000  # ROI的宽度和高度（以游戏窗口中心为中心的矩形）
roi_y_size = 1200  # ROI的宽度和高度（以游戏窗口中心为中心的矩形）
start_xy = (
    game_width // 2 - roi_x_size // 2,
    game_height // 2 - roi_y_size // 2 - 100,
)  # ROI的起始坐标 (x, y)


battle_roi_window = BaseWindow(
    start_xy[0], start_xy[1], start_xy[0] + roi_x_size, start_xy[1] + roi_y_size
)
