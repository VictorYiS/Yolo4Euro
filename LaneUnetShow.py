import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # for GPU 0,-1 for CPU.
import time
import tkinter as tk
import threading
import numpy as np
from PIL import Image, ImageTk
# from europilot.screen import stream_local_game_screen,Box
from tensorflow.keras.models import load_model
try:
    model=load_model(r'../LaneLineLableModels/run_allepoch-8935-val_loss-0.0598-val_acc-0.9794.hdf5')
except:
    model=load_model(r'LaneLineLableModels/run_allepoch-8935-val_loss-0.0598-val_acc-0.9794.hdf5')


import os
import time
from PIL import Image
import numpy as np

def stream_images_from_folder(folder_path, default_fps=1, shuffle=False, loop=False):
    """
    Generator that streams images from a folder as RGB numpy arrays.

    :param folder_path: Path to the folder containing image files.
    :param default_fps: How many images per second to yield.
    :param shuffle: Whether to shuffle the image order.
    :param loop: Whether to loop the images indefinitely.
    """
    from random import shuffle as do_shuffle

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    image_files.sort()
    if not image_files:
        raise ValueError("No valid image files found in folder.")

    if shuffle:
        do_shuffle(image_files)

    time_per_frame = 1.0 / default_fps

    while True:
        for filename in image_files:
            start = time.time()

            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)

            # Yield image and optionally receive a new FPS
            new_fps = yield image_array
            if new_fps:
                time_per_frame = 1.0 / new_fps

            elapsed = time.time() - start
            if elapsed < time_per_frame:
                time.sleep(time_per_frame - elapsed)

        if not loop:
            break




root = tk.Tk()



# Removed front_coord as we're not cropping anymore
streamer = stream_images_from_folder("detections", default_fps=25)
image_data = next(streamer)
image = Image.fromarray(image_data)
imageint = image.resize((400, 400))
imgtoest = np.array(imageint)[None, ...]
lanes = np.argmax(model(imgtoest)[0, ...], axis=-1) * 255
imglanes = Image.fromarray(lanes.astype('int8'), 'L')

# Use the full image size for the window
originalrez = imageint.size  # Using imageint size instead of cropped image
root.geometry('x'.join(str(originalrez).split(','))[1:-1].replace(' ', '') + '+1164+349')
root.title('LaneUnetShow')

tk_image = ImageTk.PhotoImage(imageint)  # Display the full image initially

# Create a label to display the image
label = tk.Label(root, image=tk_image)
label.pack()
run = 0

def update_image(setimage):
    tk_image = ImageTk.PhotoImage(setimage)
    label.configure(image=tk_image)
    label.image = tk_image
    root.update_idletasks()  # This line will update the GUI

def event_loop():
    while True:
        image_data = next(streamer)
        image = Image.fromarray(image_data)
        imageint = image.resize((400, 400))
        imgtoest = np.array(imageint)[None, ...]
        lanes = np.argmax(model(imgtoest)[0, ...], axis=-1) * 255
        imglanes = Image.fromarray(lanes.astype('int8'), 'L')
        # No resizing to originalrez needed since we're using the full image size
        update_image(setimage=imglanes)
    on_closing()

event_thread = threading.Thread(target=event_loop)
event_thread.start()

# Start the event loop
root.mainloop()
