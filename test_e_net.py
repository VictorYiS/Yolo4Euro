import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from train_e_net import ENet

#from lane_detector import ENet

# Load the pre-trained model
model_path = 'models/ENET.pth'  # Replace with the path to your trained model
enet_model = ENet(2, 4)  # Assuming you used the same model architecture

# Load the trained model's weights
enet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
enet_model.eval()  # Set the model to evaluation mode


# Define a function to process and visualize the output
def process_and_visualize(input_image_path):
    # Load and preprocess the input image
    input_image = cv2.imread(input_image_path)
    input_image = cv2.resize(input_image, (512, 256))  # Resize to the model's input size
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = input_image[..., None]
    input_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1)  # Convert to tensor

    # Pass the input image through the model
    with torch.no_grad():
        binary_logits, instance_logits = enet_model(input_tensor.unsqueeze(0))

    # Post-process the model's output
    binary_seg = torch.argmax(binary_logits, dim=1).squeeze().numpy()
    instance_seg = torch.argmax(instance_logits, dim=1).squeeze().numpy()

    # Visualize the results
    plt.figure(figsize=(12, 6))

    # Plot the input image
    plt.subplot(1, 3, 1)
    plt.imshow(input_image.squeeze(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    # Plot the binary segmentation
    plt.subplot(1, 3, 2)
    plt.imshow(binary_seg, cmap='gray')
    plt.title('Binary Segmentation')
    plt.axis('off')
    plt.show()


# Replace 'input_image.jpg' with the path to your test image
input_image_path = 'debug_images/ets2_20250518_115503_00.png'
process_and_visualize(input_image_path)
