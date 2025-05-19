import truck_telemetry

class DataLoader:
    def __init__(self):
        truck_telemetry.init()

    def get_data(self):
        data = truck_telemetry.get_data()
        if data:
            return {
                "speed": round(data["speed"] * 3.6, 3),
                "gear": data["gear"],
                "limitspeed": round(data["speedLimit"] * 3.6, 3),
                "fuel": round(data["fuel"] * 3.6, 3),
                "userSteer": data["userSteer"],
                "gameSteer": round(data["gameSteer"], 3),
                "distance": 0,
                "time": 0,
            }
        else:
            return None