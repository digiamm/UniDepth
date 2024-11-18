import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from unidepth.models import UniDepthV1

# initialize the model
model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")  # change model as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict_and_save_depth(input_folder, output_folder, visualize=False):
    """
    Predicts depth for all images in the input folder and saves the results in .npy format.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to save depth predictions.
        visualize (bool): Whether to visualize the depth prediction.
    """
    # create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    if visualize:
        # set up a single figure for dynamic updates
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax_img, ax_depth = axes
        ax_img.set_title("RGB")
        ax_img.axis("off")
        ax_depth.set_title("depth")
        ax_depth.axis("off")
        plt.ion()

    # iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        # build file paths
        input_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(input_path) or not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue  # skip non-image files

        # load and preprocess the image
        rgb_image = np.array(Image.open(input_path))
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float()  # add batch dimension (B, C, H, W)
        rgb_tensor = rgb_tensor.to(device)

        # perform inference
        with torch.no_grad():
            predictions = model.infer(rgb_tensor)
        depth = predictions["depth"].cpu().numpy().squeeze()  # remove batch dimension

        # save depth map as .npy
        output_file_name = os.path.splitext(file_name)[0] + ".npy"
        output_path = os.path.join(output_folder, output_file_name)
        np.save(output_path, depth)

        # optional visualization
        if visualize:
            ax_img.imshow(rgb_image)
            ax_depth.imshow(depth, cmap='viridis')
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)  # pause for 1 second to view each plot
        
        print(f"processed {file_name} -> saved to {output_path}")

    if visualize:
        plt.ioff()
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Depth estimation using UniDepth")
    parser.add_argument("--input-folder", type=str, required=True, help="Path to the input folder containing images")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the folder to save depth maps")
    parser.add_argument("--visualize", action="store_true", help="Visualize the depth predictions")
    args = parser.parse_args()

    predict_and_save_depth(args.input_folder, args.output_folder, args.visualize)

if __name__ == "__main__":
    main()
