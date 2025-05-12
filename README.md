# Yolo4Euro
### Yolo4Euro 使用YOLO和U-net分别做object识别和lane segmentation，然后将两者结合，得到可行路径的结果。根据结果对于EuroTruckSimulator2进行自动驾驶。
### 1. YOLOv11
使用Kitti的2d数据库进行的object检测，将训练好的模型放在"runs/detect/kitti_yolo11/weights/best.pt"
### 2. U-net
使用Kitti的lane数据库进行的lane segmentation，将训练好的模型放在"models/lane_detection_model.pth"

### 3. 结合
打开欧卡2，调到特定视角，启动脚本，按下Z切换自动驾驶的开关