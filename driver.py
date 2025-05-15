from log import log

class TruckController():
    def __init__(self):
        self.autodrive_active = True
        print("TruckController initialized")

    def get_drive_status(self):
        return self.autodrive_active

    def drive_mode_toggle(self):
        self.autodrive_active = not self.autodrive_active

    #根据车道和事件的综合因素，决定应该进行的操作
    def get_action(self, status):
        # 暂时不进行处理，只输出图像
        log.debug("TruckController: get_action called")
        return "No action needed"
