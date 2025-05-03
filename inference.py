import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import color  # Import the whole color module
from skimage.io import imread, imsave
import random
import json
from tqdm.auto import tqdm  # Use tqdm.auto for better compatibility
import sys

# Add the repository root to the Python path to ensure proper imports
sys.path.insert(0, os.path.abspath('.'))

# Import project modules
from src.prediction import Predictor
from src.gan_utils import calculate_metrics

def load_image(image_path):
    """Load an image from path and convert to RGB if grayscale"""
    img = imread(image_path)
    
    # Handle grayscale images (convert to RGB)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    
    # Normalize to [0, 1] if needed
    if img.max() > 1.0:
        img = img / 255.0
        
    return img

def main():
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the predictor with the trained model
    model_path = "models/final_generator.pth"
    print(f"Loading model from {model_path}")
    predictor = Predictor(model_path, model_type='gan')
    
    # Get list of images from data directory
    data_dir = "data/images"
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Randomly select at least 16 images
    num_images = max(16, min(len(image_files), 20))
    selected_images = random.sample(image_files, num_images)
    
    # Process each image
    all_metrics = []
    for i, img_file in enumerate(tqdm(selected_images, desc="Processing images")):
        img_path = os.path.join(data_dir, img_file)
        img_rgb = load_image(img_path)
        
        # Extract grayscale (L channel) from the image
        img_lab = color.rgb2lab(img_rgb)
        L = img_lab[:, :, 0]
        
        # Use predictor to generate colorized image
        colorized = predictor.predict(L)
        
        # Handle batch dimension if present
        if len(colorized.shape) == 4:
            colorized = colorized[0]
        
        # Calculate metrics
        psnr, ssim = calculate_metrics(colorized, img_rgb)
        all_metrics.append({
            "file": img_file,
            "psnr": float(psnr),
            "ssim": float(ssim)
        })
        
        # Create grayscale visualization
        grayscale = np.stack([L/100.0] * 3, axis=2)  # Normalize to [0,1] for visualization
        
        # Save grayscale input
        imsave(os.path.join(results_dir, f"input_{i:02d}.png"), grayscale)
        
        # Save colorized output
        imsave(os.path.join(results_dir, f"pred_{i:02d}.png"), colorized)
        
        # Save side-by-side comparison
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(grayscale)
        ax1.set_title("Grayscale Input")
        ax1.axis('off')
        
        ax2.imshow(colorized)
        ax2.set_title("GAN Colorized")
        ax2.axis('off')
        
        ax3.imshow(img_rgb)
        ax3.set_title("Original RGB")
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"comparison_{i:02d}.png"))
        plt.close()
    
    # Calculate and print overall metrics
    psnr_values = [m["psnr"] for m in all_metrics]
    ssim_values = [m["ssim"] for m in all_metrics]
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    std_psnr = np.std(psnr_values)
    std_ssim = np.std(ssim_values)
    
    print(f"\nEvaluation Results:")
    print(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    
    # Find best and worst predictions
    sorted_by_psnr = sorted(all_metrics, key=lambda x: x["psnr"], reverse=True)
    
    print("\nBest 3 colorizations by PSNR:")
    for i, m in enumerate(sorted_by_psnr[:3]):
        print(f"  {i+1}. {m['file']}: PSNR = {m['psnr']:.2f}, SSIM = {m['ssim']:.4f}")
    
    print("\nWorst 3 colorizations by PSNR:")
    for i, m in enumerate(sorted_by_psnr[-3:]):
        print(f"  {i+1}. {m['file']}: PSNR = {m['psnr']:.2f}, SSIM = {m['ssim']:.4f}")
    
    # Save metrics to file
    metrics_summary = {
        "average_psnr": float(avg_psnr),
        "std_psnr": float(std_psnr),
        "average_ssim": float(avg_ssim),
        "std_ssim": float(std_ssim),
        "best_images": sorted_by_psnr[:3],
        "worst_images": sorted_by_psnr[-3:],
        "all_metrics": all_metrics
    }
    
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/")

if __name__ == "__main__":
    main() 