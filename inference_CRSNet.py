import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
import json
import random

from model import CSRNet

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='CSRNet Batch Inference on Test Set')
parser.add_argument('--model_path', type=str,
                    help='Path to the trained .pth.tar model file')
parser.add_argument('--test_json', type=str,
                    help='Path to the test.json file containing image paths')
parser.add_argument('--output_dir', type=str, default='./test_visualizations',
                    help='Directory to save the 10 random visualization images')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU id to use. Use "cpu" for CPU.')

args = parser.parse_args()


def main():
    # --- Device Setup ---
    if args.gpu.lower() != 'cpu' and torch.cuda.is_available():
        device = torch.device('cuda:' + args.gpu)
        print(f"--- Using GPU: {args.gpu} ---")
    else:
        device = torch.device('cpu')
        print("--- Using CPU ---")

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Visualization images will be saved to: {args.output_dir}")

    # --- Model Loading ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    print(f"Loading model from {args.model_path}")
    model = CSRNet().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set model to evaluation mode

    # --- Image List Loading ---
    if not os.path.exists(args.test_json):
        print(f"Error: Test JSON file not found at {args.test_json}")
        return
    with open(args.test_json, 'r') as f:
        image_paths = json.load(f)
    print(f"Found {len(image_paths)} images to process.")

    # --- Image Pre-processing ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Batch Inference ---
    results = []
    print("\n--- Starting Inference on All Test Images ---")
    for i, img_path in enumerate(image_paths):
        img_raw = Image.open(img_path).convert('RGB')
        img_tensor = transform(img_raw).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)

        predicted_count = output.detach().cpu().sum().numpy()

        # Print real-time results
        print(
            f"[{i + 1}/{len(image_paths)}] Image: {os.path.basename(img_path)}, Predicted Count: {predicted_count:.2f}")

        # Store results for visualization later
        results.append({
            'path': img_path,
            'count': predicted_count,
            'density_map': output.detach().cpu().squeeze().numpy()
        })
    print("--- Inference Complete ---")

    # --- Visualization of Random Samples ---
    num_samples = min(10, len(results))
    if num_samples == 0:
        print("No images were processed, skipping visualization.")
        return

    print(f"\n--- Generating {num_samples} Random Visualization Samples ---")
    random_samples = random.sample(results, num_samples)

    for sample in random_samples:
        img_raw = Image.open(sample['path']).convert('RGB')
        predicted_count = sample['count']
        density_map = sample['density_map']

        # Create plot
        plt.figure(figsize=(20, 12))
        plt.subplot(1, 2, 1)
        plt.imshow(img_raw)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(density_map, cmap=cm.jet)
        plt.title(f'Predicted Density Map\n(Count: {predicted_count:.2f})')
        plt.axis('off')

        plt.tight_layout()

        # Save plot
        output_filename = os.path.join(args.output_dir, f"vis_{os.path.basename(sample['path'])}")
        plt.savefig(output_filename)
        plt.close()  # Close the figure to free up memory
        print(f"  > Saved visualization for {os.path.basename(sample['path'])}")

    print(f"--- Visualization Complete ---")


if __name__ == '__main__':
    main()
