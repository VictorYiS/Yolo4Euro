# window.py
import os
import re

import cv2
import numpy as np
import pytesseract

from log import log


# base window
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
        check similarity with a template image using OpenCV's matchTemplate.
        """
        if self.gray is None:
            print("No grayscale data to compare.")
            return False, 0.0

        template_image = BaseWindow.load_template_once(template_image_path)

        result = cv2.matchTemplate(self.gray, template_image, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, _ = cv2.minMaxLoc(result)

        return max_val >= threshold, max_val

    def __repr__(self):
        return f"BaseWindow(sx={self.sx}, sy={self.sy}, ex={self.ex}, ey={self.ey}, offset_x={BaseWindow.offset_x}, offset_y={BaseWindow.offset_y})"


class StatusWindow(BaseWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.status = 0

    def update(self):
        super().update()
        if self.color is not None:
            self.process_color()
        else:
            self.status = 0

    def process_color(self):
        pass

    def get_status(self):
        return self.status


# GearWindow class (A1-A12+, N, R1-R3)
class GearWindow(BaseWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.gear_text = "N"  # default to neutral
        self.gear_type = "neutral"  # "advance", "neutral", or "reverse"
        self.gear_number = 0
        self.gray = None

    def update(self):
        super().update()
        if self.color is not None:
            self.process_color()

    def get_status(self):
        return self.gear_text

    def process_color(self):
        if self.color is None:
            log.debug("GearWindow: No color data to process.")

        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(self.gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ARN123456789'
            text = pytesseract.image_to_string(binary, config=custom_config).strip()

            self._process_gear_text(text)

        except Exception as e:
            print(f"Gear OCR Error: {e}")
            self.reset_gear()

    def _process_gear_text(self, text):
        text = re.sub(r'\s+', '', text).upper()
        # log.debug(f"GearWindow: Processed text: {text}")

        advance_match = re.search(r'A(\d+)', text)
        reverse_match = re.search(r'R([123])', text)

        if text == 'N':
            self.gear_text = 'N'
            self.gear_type = 'neutral'
            self.gear_number = 0
        elif advance_match:
            gear_num = int(advance_match.group(1))
            if 1 <= gear_num <= 15:
                self.gear_text = f'A{gear_num}'
                self.gear_type = 'advance'
                self.gear_number = gear_num

        elif reverse_match:
            gear_num = int(reverse_match.group(1))
            if 1 <= gear_num <= 3:
                self.gear_text = f'R{gear_num}'
                self.gear_type = 'reverse'
                self.gear_number = gear_num

        else:
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
        self.gear_text = "N"
        self.gear_type = "neutral"
        self.gear_number = 0

    def __repr__(self):
        return f"GearWindow(gear={self.gear_text}, type={self.gear_type}, number={self.gear_number})"


# NumberWindow class. for recognizing numbers in a specific region
class NumberWindow(StatusWindow):
    basedir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(basedir, "..", "save")
    _save_counter = 0
    def __init__(self, sx, sy, ex, ey, min_value=0, max_value=100, open_log=False):
        super().__init__(sx, sy, ex, ey)
        self.value = 0
        self.min_value = min_value
        self.max_value = max_value
        self.gray = None
        self.open_log = open_log

        os.makedirs(self.save_dir, exist_ok=True)

    def process_color(self):
        if self.color is None:
            self.value = 0
            self.status = 0
            return

        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        # filename = os.path.join(
        #     self.save_dir,
        #     f"{self.__class__.__name__}_{type(self)._save_counter:06d}.png"
        # )
        # success = cv2.imwrite(filename, self.gray)
        # print(f"[NumberWindow] saving to {filename} →", "OK" if success else "FAILED")
        # type(self)._save_counter += 1
        resized = cv2.resize(self.gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        binary = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        try:
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
            text = pytesseract.image_to_string(binary, config=custom_config).strip()

            if self.open_log:
                print(f"[NumberWindow] OCR result: {text}")
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if numbers:
                # if multiple numbers are found, take the longest one
                longest_num = max(numbers, key=len)
                try:
                    self.value = float(longest_num)
                    self.value = max(self.min_value, min(self.max_value, self.value))
                    self.status = self.value
                except ValueError:
                    self.value = 0
                    self.status = 0
            # else:
            #     log.debug("NumberWindow: No valid number found.")

            if self.value == 0 and self.min_value > 0:
                self._try_template_matching()

        except Exception as e:
            print(f"OCR Error: {e}")
            self.value = 0
            self.status = 0

    def _try_template_matching(self):
        pass

    def get_value(self):
        return self.value

    def get_status(self):
        return self.value

    def __repr__(self):
        return f"NumberWindow(value={self.value})"



# TimeWindow class，for recognizing time in HH:MM format
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

        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

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

                if 0 <= self.hours <= 23 and 0 <= self.minutes <= 59:
                    self.update_time_values()
                else:
                    self.reset_time()
            else:
                self._try_separate_recognition(binary)
        except Exception as e:
            print(f"Time OCR Error: {e}")
            self.reset_time()

    def _try_separate_recognition(self, binary):
        height, width = binary.shape

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

                if 0 <= self.hours <= 23 and 0 <= self.minutes <= 59:
                    self.update_time_values()
                else:
                    self.reset_time()
            else:
                self.reset_time()
        except Exception:
            self.reset_time()

    def reset_time(self):
        self.hours = 0
        self.minutes = 0
        self.total_minutes = 0
        self.status = 0
        self.value = 0

    def update_time_values(self):
        self.total_minutes = self.hours * 60 + self.minutes
        self.status = self.total_minutes
        self.value = self.total_minutes

    def get_time_string(self):
        return f"{self.hours:02d}:{self.minutes:02d}"

    def get_total_minutes(self):
        return self.total_minutes

    def get_value(self):
        return self.total_minutes

    def get_status(self):
        return self.total_minutes

    def __repr__(self):
        return f"TimeWindow(time={self.get_time_string()}, minutes={self.total_minutes})"


# find the game window logo
def find_game_window_logo(frame, template_path, threshold):
    return (0, 0)
    # read the template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Failed to load template image from {template_path}")
        return None

    # turn frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # if threshold is not None:
    if max_val >= threshold:
        return max_loc
    else:
        return None


def set_windows_offset(frame):
    # find the game window logo
    logo_position = find_game_window_logo(frame, "./images/title_logo.png", 0.8)

    if logo_position is not None:
        offset_x, offset_y = logo_position

        offset_x += 5 # adjust for title bar
        offset_y += 45

        # offset_x
        BaseWindow.set_offset(offset_x, offset_y)
        BaseWindow.set_frame(frame)
        BaseWindow.update_all()

        print(f"All windows offset by ({offset_x}, {offset_y})")
        return True
    else:
        print("Failed to find the game logo, offsets not set.")
        return False


# adjust game window size and scale
game_width = 1920
game_height = round(game_width * 0.5625)

base_width = 1920 # do not change
base_height = 1080 # do not change

width_scale = game_width / base_width
height_scale = game_height / base_height

print(f"width_scale: {width_scale}, height_scale: {height_scale}")


def convert_coordinates(x1, y1, x2, y2):
    # return x1, y1, x2, y2
    new_x1 = round(x1 * width_scale)
    new_y1 = round(y1 * height_scale)
    new_x2 = round(x2 * width_scale)
    new_y2 = round(y2 * height_scale)
    return new_x1, new_y1, new_x2, new_y2


game_window = BaseWindow(0, 0, game_width, game_height)

# # OCR for self speed, distance, time, set speed, gear
# self_speed_window = NumberWindow(*convert_coordinates(1486, 706, 1513, 728), min_value=0, max_value=90, open_log=True)
# self_distance_window = NumberWindow(*convert_coordinates(1590, 735, 1620, 757))
# self_time_window = TimeWindow(1831, 707, 1884, 727)
# self_set_speed = NumberWindow(*convert_coordinates(1498,787,1520,803))
# self_gear_window = GearWindow(1601, 700, 1625, 729)

roi_x_size = 1000  # ROI width and height (rectangle centered on the game window)
roi_y_size = 500
start_xy = (
    game_width // 2 - roi_x_size // 2 + 50,
    game_height // 2 - roi_y_size // 2 - 50,
)  # start coordinates for the ROI


battle_roi_window = BaseWindow(
    start_xy[0], start_xy[1], start_xy[0] + roi_x_size, start_xy[1] + roi_y_size
)
