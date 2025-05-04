import cv2
import os
from tqdm import tqdm
import glob


def video_to_frames(video_path, output_dir, fps=10):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率：{video_fps} FPS, 总帧数：{total_frames}")
    interval = int(video_fps / fps) if fps else 1

    count = 0
    saved = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    with tqdm(total=total_frames, desc=f"提取：{video_name}", unit="帧") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                filename = os.path.join(output_dir, f"{video_name}_frame_{saved:06d}.jpg")
                cv2.imwrite(filename, frame)
                saved += 1
            count += 1
            pbar.update(1)

    cap.release()
    print(f"✅ {video_name} 提取完成，共 {saved} 帧")


def batch_process_all_videos(input_folder, output_folder, fps=10):
    os.makedirs(output_folder, exist_ok=True)
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))

    for video_file in video_files:
        print(f"正在处理视频：{video_file}")
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_dir = os.path.join(output_folder, video_name)
        video_to_frames(video_file, output_dir, fps=fps)


batch_process_all_videos("./videos", "output_frames", fps=24)
