# grabscreen.py
import numpy as np
import atexit
from log import log
import time
import mss
import cv2

# Global screen capture object
sct = None


def init_camera(target_fps):
    global sct
    if sct is None:
        log.debug("Initializing Screen Capture with MSS.\n")
        sct = mss.mss()
        atexit.register(sct.close)
        time.sleep(1)  # Short wait for initialization


def grab_screen():
    """Capture the screen using MSS instead of dxcam"""
    global sct

    try:
        # Capture entire primary monitor
        monitor = sct.monitors[1]  # Primary monitor

        # Get screenshot
        screenshot = sct.grab(monitor)

        # Convert to numpy array
        frame = np.array(screenshot)

        # Convert BGRA to BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame
    except Exception as e:
        log.error(f"Error capturing screen: {e}")
        return None