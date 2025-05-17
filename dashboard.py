# Simplified truck dashboard
import grabscreen
from detector import YOLODetector, LaneDetector
from window import *


class TruckDashboard:
    def __init__(self):
        super().__init__()
        self.speed = 0
        self.distance = 0
        self.time = 0
        # self.gear = 0
        self.fuel = 100.0
        # self.yolo_detector = YOLODetector("models/1000m_736sgz.pt")
        self.vehicle_detector = YOLODetector("models/best.pt")
        self.lane_mask_detector = LaneDetector()
        self.lane_status = None
        self.classes = []

    def get_stored_data(self):
        return {
            "speed": self.speed,
            "distance": self.distance,
            "time": self.time,
            # "gear": self.gear,
            "fuel": self.fuel,
            "lane_status": self.lane_status,
            "classes": self.classes,
            "detect_frame": self.detect_frame,
        }

    def update_data(self):
        frame = grabscreen.grab_screen()
        BaseWindow.set_frame(frame)
        BaseWindow.update_all()
        self.speed = self_speed_window.get_status()
        self.distance = self_distance_window.get_status()
        self.time = self_time_window.get_status()
        # self.gear = gear
        # self.fuel = self_fuel_window.get_status()

        roi = game_window.color
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)
        self.lane_status = self.lane_mask_detector(roi)

        # # 车道线检测
        # self.lane_status = self.lane_mask_detector.detect(roi)
        # if self.lane_status[1] is None or not np.any(self.lane_status[1]):
        #     print("Warning: Lane mask is empty or all zeros")


        yolo_results = self.vehicle_detector.detect(roi_rgb)
        if yolo_results is not None:
            objects, self.classes = self.vehicle_detector.process_detections(roi_rgb, yolo_results)
        else:
            print("No detection results.")
        self.detect_frame = roi_rgb

    def get_frame_and_status(self):
        # frame = game_window.color
        status = self.get_stored_data()
        return status
