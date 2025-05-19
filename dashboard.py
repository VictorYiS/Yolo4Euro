# Simplified truck dashboard
import time

import grabscreen
from data_loader import DataLoader
from detector import YOLODetector, LaneDetector
from window import *


class TruckDashboard:
    def __init__(self):
        super().__init__()
        self.speed = 0
        self.setspeed = 0
        self.distance = 0
        self.time = 0
        self.gear = 0
        self.fuel = 100
        self.user_steer = 0
        self.game_steer = 0
        self.traffic_detector = YOLODetector("models/1000m_736sgz.pt")
        # self.vehicle_detector = YOLODetector("models/best.pt")
        self.lane_mask_detector = LaneDetector()
        self.data_laoder = DataLoader()
        self.lane_status = None
        # self.car_detect = None
        self.traffic_detect = None
        self.detect_frame = None

    def get_stored_data(self):
        return {
            "speed": self.speed,
            "setspeed": self.setspeed,
            "distance": self.distance,
            "time": self.time,
            "gear": self.gear,
            "fuel": self.fuel,
            "lane_status": self.lane_status,
            # "car_detect": self.car_detect,
            "traffic_detect": self.traffic_detect,
            "detect_frame": self.detect_frame,
            "user_steer": self.user_steer,
            "game_steer": self.game_steer
        }

    def update_data(self):
        frame = grabscreen.grab_screen()
        BaseWindow.set_frame(frame)
        BaseWindow.update_all()
        vehicle_data = self.data_laoder.get_data()
        self.speed = vehicle_data["speed"]
        self.distance = vehicle_data["distance"]
        self.time = vehicle_data["time"]
        self.setspeed = vehicle_data["limitspeed"]
        self.gear = vehicle_data["gear"]
        self.fuel = vehicle_data["fuel"]
        self.user_steer = vehicle_data["userSteer"]
        self.game_steer = vehicle_data["gameSteer"]

        roi = battle_roi_window.color
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)
        frame_time = time.time()
        self.lane_status = self.lane_mask_detector(roi)
        # cv2.imwrite("debug_images/roi_rgb_{}.png".format(frame_time), roi_rgb)
        # self.lane_status["lane_image"].save("debug_images/frame_{}.png".format(frame_time))

        # yolo_results = self.vehicle_detector.detect(roi_rgb)
        # if yolo_results is not None:
        #     self.car_detect, classes = self.vehicle_detector.process_detections(roi_rgb, yolo_results)
        yolo_results = self.traffic_detector.detect(roi_rgb)
        if yolo_results is not None:
            self.traffic_detect, classes = self.traffic_detector.process_detections(roi_rgb, yolo_results)
        else:
            print("No detection results.")
        self.detect_frame = frame_time

    def get_frame_and_status(self):
        # frame = game_window.color
        status = self.get_stored_data()
        return status
