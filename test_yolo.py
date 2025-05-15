import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


# 1. Load your trained model
def load_model(weights_path):
    """Load trained YOLO model from weights file"""
    model = YOLO(weights_path)
    print(f"Model loaded: {weights_path}")
    return model


# 2. Perform detection on a single image
def detect_image(model, image_path, conf_threshold=0.25):
    """Run detection on a single image"""
    # Perform prediction
    results = model(image_path, conf=conf_threshold)

    # Get results
    result = results[0]  # First image result

    # Visualize results
    img = cv2.imread(image_path)
    for box in result.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Get confidence and class
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]

        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save or display result
    output_path = os.path.join('runs/detect/predict', os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Detection result saved to: {output_path}")

    return img, result


# 3. Process a video file
def detect_video(model, video_path, conf_threshold=0.25, output_path=None):
    """Run detection on a video file"""
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output video writer
    if output_path is None:
        output_path = os.path.join('runs/detect/predict', os.path.basename(video_path))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame, conf=conf_threshold)
        result = results[0]

        # Draw detections
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete: {output_path}")

    return output_path


# 4. Integrate with EuroTrackSimulator2
def detect_realtime_game(model, conf_threshold=0.25):
    """Run real-time detection on game window - simple prototype"""
    # You would typically:
    # 1. Capture the game window (using screen capture libraries)
    # 2. Process the frame with your model
    # 3. Use detection results to control the vehicle

    try:
        import dxcam  # For game capture

        # Create screen capture object
        cam = dxcam.create()

        # Game window position (adjust to your game window)
        # You might need to detect this automatically
        region = (0, 0, 1920, 1080)  # (left, top, right, bottom)

        print("Press 'q' to quit...")

        # Main processing loop
        while True:
            # Capture frame from game
            frame = cam.grab(region=region)
            if frame is None:
                continue

            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Perform detection
            results = model(frame, conf=conf_threshold)
            result = results[0]

            # Process detections
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display frame with detections
            cv2.imshow('EuroTrackSimulator2 Detection', frame)

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    except ImportError:
        print("dxcam not installed. Please install it with: pip install dxcam")
        print("Falling back to simple webcam detection...")

        # Fallback to webcam
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = model(frame, conf=conf_threshold)
            result = results[0]

            # Draw detections
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Webcam Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Main function to use the trained model
if __name__ == "__main__":
    # Path to your trained weights
    weights_path = "runs/detect/kitti_yolo11/weights/best.pt"

    # Load model
    model = load_model(weights_path)

    # Choose detection mode
    mode = "image"  # Options: "image", "video", "realtime"

    if mode == "image":
        # Test on a single image
        test_image = "output_frames/2/2_frame_000585.jpg"
        img, result = detect_image(model, test_image, conf_threshold=0.25)

    elif mode == "video":
        # Test on a video file
        test_video = "path/to/test/video.mp4"
        detect_video(model, test_video, conf_threshold=0.25)

    elif mode == "realtime":
        # Real-time detection for game
        detect_realtime_game(model, conf_threshold=0.25)
