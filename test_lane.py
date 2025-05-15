from detector import LaneDetector

lane_detector = LaneDetector()
lane_detector.process_folder("detections/2025-05-15_17-07-13", "debug_images")
