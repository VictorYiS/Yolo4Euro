import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Add this line at the very top


# Define the UNet model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle
        self.conv3 = DoubleConv(128, 256)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x = self.pool1(x1)

        x2 = self.conv2(x)
        x = self.pool2(x2)

        # Middle
        x = self.conv3(x)

        # Decoder
        x = self.upconv1(x)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.conv4(x)

        x = self.upconv2(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv5(x)

        # Output
        x = self.out_conv(x)
        return torch.sigmoid(x)


# Custom Dataset class using only OpenCV
class RoadDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        # Resize to fixed size
        img = cv2.resize(img, (512, 256))
        mask = cv2.resize(mask, (512, 256))

        # Normalize image
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225],
                                                                                   dtype=np.float32)
        img = img.transpose(2, 0, 1)  # HWC to CHW format

        # Normalize mask to binary values
        mask = (mask > 127).astype(np.float32)

        # Convert to tensors with explicit dtype
        img_tensor = torch.from_numpy(img).float()  # Explicitly convert to float32
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension

        return img_tensor, mask_tensor


# Function to preprocess KITTI dataset
def preprocess_kitti_dataset(data_dir, output_dir):
    img_dir = os.path.join(output_dir, 'images')
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Get all training images
    train_images = sorted(glob(os.path.join(data_dir, 'training/image_2/*.png')))
    gt_images = sorted(glob(os.path.join(data_dir, 'training/gt_image_2/*.png')))

    print(f"Found {len(train_images)} training images and {len(gt_images)} ground truth images")

    for i, (img_path, gt_path) in enumerate(zip(train_images, gt_images)):
        # Load images
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)

        if img is None or gt is None:
            print(f"Warning: Couldn't load {img_path} or {gt_path}")
            continue

        # Convert ground truth to road mask
        # In KITTI, road is typically marked as pink/magenta
        hsv = cv2.cvtColor(gt, cv2.COLOR_BGR2HSV)

        # Road area (pink/magenta in KITTI)
        lower_road = np.array([120, 30, 80])
        upper_road = np.array([180, 255, 255])
        road_mask = cv2.inRange(hsv, lower_road, upper_road)

        # Resize to standard size
        img_resized = cv2.resize(img, (512, 256))
        mask_resized = cv2.resize(road_mask, (512, 256))

        # Save processed images
        img_name = os.path.basename(img_path)
        mask_name = f"mask_{img_name}"

        cv2.imwrite(os.path.join(img_dir, img_name), img_resized)
        cv2.imwrite(os.path.join(mask_dir, mask_name), mask_resized)

        # Show progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(train_images)} images")

            # Save visualization for every 10th image
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # Create visualization
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
            plt.title("Ground Truth")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(road_mask, cmap='gray')
            plt.title("Road Mask")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"vis_{img_name}"))
            plt.close()

    print(f"Preprocessing complete. Images saved to {output_dir}")
    return img_dir, mask_dir


# Train a model on the preprocessed data
def train_model(img_dir, mask_dir, model_save_path, epochs=10, batch_size=4):
    # Get all preprocessed images and masks
    images = sorted(glob(os.path.join(img_dir, '*.png')))
    masks = sorted(glob(os.path.join(mask_dir, '*.png')))

    print(f"Found {len(images)} images and {len(masks)} masks for training")

    # Create dataset and dataloader
    dataset = RoadDataset(images, masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=3, out_channels=1)
    model = model.float()  # Explicitly set model to float32
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    history = {'loss': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for inputs, masks in progress_bar:
            inputs = inputs.to(device)
            masks = masks.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate epoch loss
        epoch_loss = running_loss / len(dataset)
        history['loss'].append(epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Validate and visualize intermediate results
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            model.eval()

            # Get a sample batch
            with torch.no_grad():
                # Get first batch
                val_inputs, val_masks = next(iter(dataloader))
                val_inputs = val_inputs.to(device)
                val_outputs = model(val_inputs)

                # Convert to numpy for visualization
                val_inputs = val_inputs.cpu().numpy()
                val_masks = val_masks.cpu().numpy()
                val_outputs = val_outputs.cpu().numpy()

                # Visualize first sample
                plt.figure(figsize=(15, 5))

                # Denormalize image for display
                img = val_inputs[0].transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title("Input Image")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(val_masks[0][0], cmap='gray')
                plt.title("Ground Truth")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(val_outputs[0][0], cmap='gray')
                plt.title(f"Prediction (Epoch {epoch + 1})")
                plt.axis('off')

                plt.tight_layout()

                # Create directory for visualizations
                vis_dir = os.path.dirname(model_save_path)
                os.makedirs(vis_dir, exist_ok=True)

                plt.savefig(f"{vis_dir}/train_progress_epoch_{epoch + 1}.png")
                plt.close()

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f"{os.path.dirname(model_save_path)}/training_loss.png")
    plt.close()

    # Save model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': history['loss']
    }, model_save_path)

    print(f"Model saved to {model_save_path}")

    return model


# Test function
def test_model(model_path, test_image_path, save_path=None):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = UNet(in_channels=3, out_channels=1)
    model.float()  # Explicitly set model to float32
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load and preprocess test image
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Error: Could not load image at {test_image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save original dimensions
    orig_h, orig_w = img.shape[:2]

    # Resize and preprocess
    img_resized = cv2.resize(img, (512, 256))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = img_normalized.transpose(2, 0, 1)  # HWC to CHW

    # Convert to tensor
    input_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0).to(device)  # Explicitly convert to float32

    # Inference
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.squeeze(0).squeeze(0).cpu().numpy()

    print(f"Prediction stats - min: {prediction.min()}, max: {prediction.max()}, mean: {prediction.mean()}")

    # Visualize results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='jet')
    plt.colorbar()
    plt.title("Prediction Heatmap")
    plt.axis('off')

    # Threshold prediction and overlay
    mask = (prediction > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (orig_w, orig_h))

    color_mask = np.zeros_like(img)
    color_mask[:, :, 1] = mask_resized  # Green channel

    overlay = cv2.addWeighted(img, 1, color_mask, 0.7, 0)

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved result to {save_path}")

    plt.show()
    plt.close()

    return prediction, mask_resized


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    kitti_dataset_path = os.path.join(project_root, "data_road")
    processed_data_path = os.path.join(project_root, "processed_data")
    model_save_path = os.path.join(project_root, "models", "lane_detection_model.pth")

    # Create output directories
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Step 1: Preprocess KITTI dataset
    print("Preprocessing KITTI dataset...")
    img_dir, mask_dir = preprocess_kitti_dataset(kitti_dataset_path, processed_data_path)

    # Step 2: Train model
    print("Training road segmentation model...")
    model = train_model(img_dir, mask_dir, model_save_path, epochs=20, batch_size=4)

    # Step 3: Test on sample images
    test_dir = os.path.join(kitti_dataset_path, "testing/image_2")
    test_images = sorted(os.listdir(test_dir))[:5]  # Test on first 5 images

    print("Testing model on sample images...")
    for test_img in test_images:
        test_path = os.path.join(test_dir, test_img)
        save_path = os.path.join("../results", f"result_{test_img}")
        test_model(model_save_path, test_path, save_path)

    print("Done!")


if __name__ == "__main__":
    main()