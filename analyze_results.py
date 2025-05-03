#!/usr/bin/env python3
import os
import glob
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import json

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Create analysis directory
    analysis_dir = "analysis"
    ensure_dir(analysis_dir)
    
    # Find all generated sample images from training
    epoch_images = sorted(glob.glob("models/epoch_*_iter_*.png"))
    if not epoch_images:
        print("No training samples found in models/ directory.")
        return
    
    # Group images by epoch
    epochs = {}
    for img_path in epoch_images:
        basename = os.path.basename(img_path)
        parts = basename.split('_')
        epoch_num = int(parts[1])
        iteration = int(parts[3].split('.')[0])
        
        if epoch_num not in epochs:
            epochs[epoch_num] = []
        epochs[epoch_num].append((iteration, img_path))
    
    # Sort epochs and iterations
    sorted_epochs = sorted(epochs.keys())
    
    # Plot progress over epochs
    plt.figure(figsize=(15, 10))
    rows = min(3, len(sorted_epochs))
    cols = min(4, len(sorted_epochs))
    
    for i, epoch in enumerate(sorted_epochs):
        if i >= rows * cols:
            break
            
        # Choose the first iteration of each epoch
        img_path = sorted(epochs[epoch], key=lambda x: x[0])[0][1]
        img = imread(img_path)
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"Epoch {epoch}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "training_progress.png"))
    
    # Create a comparison between first and last epoch
    if len(sorted_epochs) >= 2:
        plt.figure(figsize=(12, 6))
        
        # First epoch
        first_epoch = sorted_epochs[0]
        first_img_path = sorted(epochs[first_epoch], key=lambda x: x[0])[0][1]
        first_img = imread(first_img_path)
        
        plt.subplot(1, 2, 1)
        plt.imshow(first_img)
        plt.title(f"Epoch {first_epoch}")
        plt.axis('off')
        
        # Last epoch
        last_epoch = sorted_epochs[-1]
        last_img_path = sorted(epochs[last_epoch], key=lambda x: x[0])[-1][1]
        last_img = imread(last_img_path)
        
        plt.subplot(1, 2, 2)
        plt.imshow(last_img)
        plt.title(f"Epoch {last_epoch}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "first_vs_last.png"))
    
    # Print analysis summary to a text file
    with open(os.path.join(analysis_dir, "training_summary.txt"), "w") as f:
        f.write("Training Progress Analysis\n")
        f.write("=========================\n\n")
        f.write(f"Total number of epochs: {len(sorted_epochs)}\n")
        f.write(f"Epochs found: {sorted_epochs}\n\n")
        
        f.write("Training Samples:\n")
        for epoch in sorted_epochs:
            iterations = [it for it, _ in epochs[epoch]]
            f.write(f"  Epoch {epoch}: {len(iterations)} samples, iterations {sorted(iterations)}\n")
        
        # Model files info
        generator_path = "models/final_generator.pth"
        discriminator_path = "models/final_discriminator.pth"
        
        f.write("\nModel Files:\n")
        if os.path.exists(generator_path):
            size_mb = os.path.getsize(generator_path) / (1024 * 1024)
            f.write(f"  Generator: {generator_path} ({size_mb:.2f} MB)\n")
        else:
            f.write("  Generator: Not found\n")
            
        if os.path.exists(discriminator_path):
            size_mb = os.path.getsize(discriminator_path) / (1024 * 1024)
            f.write(f"  Discriminator: {discriminator_path} ({size_mb:.2f} MB)\n")
        else:
            f.write("  Discriminator: Not found\n")
    
    # Create visual array of the final epoch samples
    if len(sorted_epochs) > 0:
        final_epoch = sorted_epochs[-1]
        final_samples = sorted(epochs[final_epoch], key=lambda x: x[0])
        
        num_samples = len(final_samples)
        if num_samples > 0:
            # Determine grid size
            grid_size = int(np.ceil(np.sqrt(num_samples)))
            
            plt.figure(figsize=(15, 15))
            for i, (_, img_path) in enumerate(final_samples):
                if i >= grid_size * grid_size:
                    break
                    
                plt.subplot(grid_size, grid_size, i + 1)
                img = imread(img_path)
                plt.imshow(img)
                plt.title(f"Iteration {final_samples[i][0]}")
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"epoch_{final_epoch}_samples.png"))
    
    # Look for evaluation metrics
    metrics_files = glob.glob("results/*/evaluation_metrics.json")
    if metrics_files:
        latest_metrics = max(metrics_files, key=os.path.getctime)
        
        try:
            with open(latest_metrics, 'r') as f:
                metrics = json.load(f)
                
            # Write metrics summary
            with open(os.path.join(analysis_dir, "metrics_summary.txt"), "w") as f:
                f.write("Model Performance Metrics\n")
                f.write("========================\n\n")
                
                if 'average_psnr' in metrics:
                    f.write(f"Average PSNR: {metrics['average_psnr']:.2f}")
                    if 'std_psnr' in metrics:
                        f.write(f" ± {metrics['std_psnr']:.2f}\n")
                    else:
                        f.write("\n")
                        
                if 'average_ssim' in metrics:
                    f.write(f"Average SSIM: {metrics['average_ssim']:.4f}")
                    if 'std_ssim' in metrics:
                        f.write(f" ± {metrics['std_ssim']:.4f}\n")
                    else:
                        f.write("\n")
                
                # Best and worst images
                if 'best_images' in metrics:
                    f.write("\nBest Images by PSNR:\n")
                    for i, img in enumerate(metrics['best_images']):
                        f.write(f"  {i+1}. {img['file']}: PSNR = {img['psnr']:.2f}, SSIM = {img['ssim']:.4f}\n")
                
                if 'worst_images' in metrics:
                    f.write("\nWorst Images by PSNR:\n")
                    for i, img in enumerate(metrics['worst_images']):
                        f.write(f"  {i+1}. {img['file']}: PSNR = {img['psnr']:.2f}, SSIM = {img['ssim']:.4f}\n")
        except:
            print(f"Error processing metrics file: {latest_metrics}")
    
    print(f"Analysis results saved to {analysis_dir}/")
    print("GAN training analysis complete!")

if __name__ == "__main__":
    main() 