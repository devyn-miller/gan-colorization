#!/usr/bin/env python3
"""
GAN-based Image Colorization Evaluation Script

This script performs comprehensive evaluation and visualization for a GAN-based 
grayscale-to-color image colorization system. It analyzes results from the most
recent training run, computes various image quality metrics, and generates 
visualization plots for easy interpretation of model performance.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
from datetime import datetime
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import lpips

# Configuration
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_FILE = "evaluation_metrics.json"
SUMMARY_FILE = "evaluation_summary.csv"
TOP_N = 5  # Number of best/worst images to highlight

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_latest_run():
    """Find the most recent results folder"""
    result_dirs = glob.glob(os.path.join(RESULTS_DIR, "run_*"))
    if not result_dirs:
        raise FileNotFoundError(f"No result directories found in '{RESULTS_DIR}'")
    
    # Sort by creation time (most recent first)
    latest_dir = max(result_dirs, key=os.path.getctime)
    print(f"Using most recent results from: {latest_dir}")
    return latest_dir

def load_images(result_dir):
    """Load all result images from the specified directory"""
    # Find all image types
    comparison_images = sorted(glob.glob(os.path.join(result_dir, "comparison_*.png")))
    grayscale_images = sorted(glob.glob(os.path.join(result_dir, "grayscale_*.png")))
    original_images = sorted(glob.glob(os.path.join(result_dir, "original_*.png")))
    
    # Load the images
    comparisons = [imread(img) for img in comparison_images]
    grayscales = [imread(img) for img in grayscale_images]
    originals = [imread(img) for img in original_images]
    
    print(f"Loaded {len(comparisons)} comparison images")
    print(f"Loaded {len(grayscales)} grayscale images")
    print(f"Loaded {len(originals)} original images")
    
    if comparison_images:
        print(f"Sample filenames: {os.path.basename(comparison_images[0])}, {os.path.basename(grayscale_images[0])}")
    
    return {
        'comparison_paths': comparison_images,
        'grayscale_paths': grayscale_images,
        'original_paths': original_images,
        'comparisons': comparisons,
        'grayscales': grayscales,
        'originals': originals
    }

def extract_colorization_from_comparison(comparison_image, target_image):
    """Extract the colorized part from a comparison image and resize to match target dimensions"""
    # This function extracts the colorized result from the comparison image
    h, w, _ = comparison_image.shape
    
    # Assuming the colorized result is in the middle third of the comparison image
    colorized = comparison_image[:, w//3:2*w//3, :]
    
    # Resize to match target image dimensions if needed
    target_h, target_w, _ = target_image.shape
    if colorized.shape[0] != target_h or colorized.shape[1] != target_w:
        colorized = cv2.resize(colorized, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    return colorized

def compute_metrics(images):
    """Compute various image quality metrics between original and colorized images"""
    results = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    deltaE_values = []
    hist_corr_values = []
    
    # Initialize LPIPS
    lpips_fn = lpips.LPIPS(net='alex')
    
    for i, (comparison, original) in enumerate(zip(images['comparisons'], images['originals'])):
        # Extract colorized image from comparison (resize to match original)
        colorized = extract_colorization_from_comparison(comparison, original)
        
        # Compute PSNR (Peak Signal-to-Noise Ratio)
        psnr_val = psnr(original, colorized)
        psnr_values.append(psnr_val)
        
        # Compute SSIM (Structural Similarity Index)
        ssim_val = ssim(original, colorized, channel_axis=2, data_range=1.0)
        ssim_values.append(ssim_val)
        
        # Compute LPIPS (Learned Perceptual Image Patch Similarity)
        # Convert images to PyTorch tensors (normalized to [-1, 1])
        try:
            # Make sure both tensors are 3-channel RGB images
            if original.shape[2] != 3 or colorized.shape[2] != 3:
                print(f"Warning: Non-RGB image detected. Original shape: {original.shape}, Colorized shape: {colorized.shape}")
                # Skip LPIPS for this image
                lpips_val = float('nan')
            else:
                img1 = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
                img2 = torch.from_numpy(colorized).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
                lpips_val = lpips_fn(img1, img2).item()
        except Exception as e:
            print(f"LPIPS computation error: {e}")
            lpips_val = float('nan')  # Use NaN as fallback
            
        lpips_values.append(lpips_val)
        
        # Compute Delta E (color difference)
        # Sample a few points for efficiency
        try:
            if original.shape[2] != 3 or colorized.shape[2] != 3:
                print(f"Warning: Skipping Delta E for non-RGB image")
                deltaE_val = float('nan')
            else:
                h, w, _ = original.shape
                deltaE_samples = []
                for _ in range(100):  # Sample 100 random pixels
                    y, x = np.random.randint(0, h), np.random.randint(0, w)
                    rgb1 = sRGBColor(original[y, x, 0], original[y, x, 1], original[y, x, 2], is_upscaled=True)
                    rgb2 = sRGBColor(colorized[y, x, 0], colorized[y, x, 1], colorized[y, x, 2], is_upscaled=True)
                    lab1 = convert_color(rgb1, LabColor)
                    lab2 = convert_color(rgb2, LabColor)
                    try:
                        # Handle potential numpy.asscalar deprecation
                        delta_e_result = delta_e_cie2000(lab1, lab2)
                        # Convert numpy array to scalar if needed
                        if hasattr(delta_e_result, 'item'):
                            delta_e_result = delta_e_result.item()
                        elif hasattr(delta_e_result, '__iter__'):
                            delta_e_result = float(delta_e_result)
                        deltaE_samples.append(delta_e_result)
                    except Exception as e:
                        print(f"Delta E calculation error: {e}")
                        continue
                        
                deltaE_val = np.mean(deltaE_samples) if deltaE_samples else float('nan')
        except Exception as e:
            print(f"Delta E computation error: {e}")
            deltaE_val = float('nan')
            
        deltaE_values.append(deltaE_val)
        
        # Compute histogram correlation
        try:
            hist_corr = 0
            for ch in range(min(3, original.shape[2])):  # For each RGB channel (up to 3)
                hist1 = cv2.calcHist([original], [ch], None, [256], [0, 256])
                hist2 = cv2.calcHist([colorized], [ch], None, [256], [0, 256])
                hist_corr += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            hist_corr /= min(3, original.shape[2])  # Average across channels
            hist_corr_values.append(hist_corr)
        except Exception as e:
            print(f"Histogram correlation error: {e}")
            hist_corr_values.append(float('nan'))
        
        # Store individual image results
        results.append({
            'image_id': i,
            'filename': os.path.basename(images['comparison_paths'][i]),
            'psnr': psnr_val,
            'ssim': ssim_val,
            'lpips': lpips_val,
            'deltaE': deltaE_val,
            'hist_corr': hist_corr
        })
    
    # Filter out NaN values for statistics calculation
    valid_lpips = [v for v in lpips_values if not np.isnan(v)]
    valid_deltaE = [v for v in deltaE_values if not np.isnan(v)]
    valid_hist_corr = [v for v in hist_corr_values if not np.isnan(v)]
    
    # Calculate overall statistics
    metrics = {
        'psnr': np.mean(psnr_values),
        'psnr_std': np.std(psnr_values),
        'ssim': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'lpips': np.mean(valid_lpips) if valid_lpips else float('nan'),
        'lpips_std': np.std(valid_lpips) if valid_lpips else float('nan'),
        'deltaE': np.mean(valid_deltaE) if valid_deltaE else float('nan'),
        'deltaE_std': np.std(valid_deltaE) if valid_deltaE else float('nan'),
        'hist_corr': np.mean(valid_hist_corr) if valid_hist_corr else float('nan'),
        'hist_corr_std': np.std(valid_hist_corr) if valid_hist_corr else float('nan'),
        'individual_results': results
    }
    
    return metrics

def load_or_compute_metrics(images, result_dir):
    """Load metrics from file or compute them if not available"""
    metrics_path = os.path.join(result_dir, METRICS_FILE)
    
    # Try to load metrics from file
    if os.path.exists(metrics_path):
        print(f"Loading metrics from {metrics_path}")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Check if we need to compute individual results
        if 'individual_results' not in metrics:
            print("Computing individual image metrics...")
            full_metrics = compute_metrics(images)
            metrics['individual_results'] = full_metrics['individual_results']
    else:
        # Fall back to the root metrics file
        root_metrics_path = METRICS_FILE
        if os.path.exists(root_metrics_path):
            print(f"Loading metrics from {root_metrics_path}")
            with open(root_metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Compute individual results
            print("Computing individual image metrics...")
            full_metrics = compute_metrics(images)
            metrics['individual_results'] = full_metrics['individual_results']
        else:
            # Compute everything from scratch
            print("Computing all metrics from scratch...")
            metrics = compute_metrics(images)
    
    return metrics

def create_metric_plots(metrics, output_dir):
    """Create and save plots visualizing the metrics"""
    ensure_dir(output_dir)
    
    # Extract individual results for plotting
    results = metrics['individual_results']
    df = pd.DataFrame(results)
    
    # 1. Distribution of PSNR
    plt.figure(figsize=(10, 6))
    sns.histplot(df['psnr'], kde=True)
    plt.title(f'PSNR Distribution: {metrics["psnr"]:.2f} ± {metrics["psnr_std"]:.2f}')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_distribution.png'))
    plt.close()
    
    # 2. Distribution of SSIM
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ssim'], kde=True)
    plt.title(f'SSIM Distribution: {metrics["ssim"]:.4f} ± {metrics["ssim_std"]:.4f}')
    plt.xlabel('SSIM')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_distribution.png'))
    plt.close()
    
    # 3. PSNR vs SSIM scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='psnr', y='ssim', data=df)
    plt.title('PSNR vs SSIM')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('SSIM')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_vs_ssim.png'))
    plt.close()
    
    # 4. Bar plot of best PSNR values
    best_psnr = df.nlargest(TOP_N, 'psnr')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='psnr', y='filename', data=best_psnr)
    plt.title(f'Top {TOP_N} Images by PSNR')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Image')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_psnr.png'))
    plt.close()
    
    # 5. Bar plot of worst PSNR values
    worst_psnr = df.nsmallest(TOP_N, 'psnr')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='psnr', y='filename', data=worst_psnr)
    plt.title(f'Bottom {TOP_N} Images by PSNR')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Image')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'worst_psnr.png'))
    plt.close()
    
    # 6. Delta E distribution
    if 'deltaE' in metrics and not np.isnan(metrics['deltaE']) and 'deltaE' in df.columns:
        # Filter out NaN values
        valid_deltaE = df[~np.isnan(df['deltaE'])]
        if not valid_deltaE.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(valid_deltaE['deltaE'], kde=True)
            plt.title(f'ΔE Distribution: {metrics["deltaE"]:.2f} ± {metrics["deltaE_std"]:.2f}')
            plt.xlabel('ΔE (CIEDE2000)')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'deltaE_distribution.png'))
            plt.close()
    
    # 7. Histogram correlation distribution
    if 'hist_corr' in metrics and not np.isnan(metrics['hist_corr']) and 'hist_corr' in df.columns:
        # Filter out NaN values
        valid_hist_corr = df[~np.isnan(df['hist_corr'])]
        if not valid_hist_corr.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(valid_hist_corr['hist_corr'], kde=True)
            plt.title(f'Histogram Correlation: {metrics["hist_corr"]:.4f} ± {metrics["hist_corr_std"]:.4f}')
            plt.xlabel('Correlation')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hist_corr_distribution.png'))
            plt.close()
    
    print(f"Saved all plots to {output_dir}")

def highlight_best_worst(metrics, images, output_dir):
    """Create visual highlights of the best and worst performing images"""
    results = metrics['individual_results']
    df = pd.DataFrame(results)
    
    # Create the highlight directory
    highlight_dir = os.path.join(output_dir, 'highlights')
    ensure_dir(highlight_dir)
    
    # Get best and worst by PSNR
    best_psnr_idx = df.nlargest(TOP_N, 'psnr')['image_id'].values
    worst_psnr_idx = df.nsmallest(TOP_N, 'psnr')['image_id'].values
    
    # Get best and worst by SSIM
    best_ssim_idx = df.nlargest(TOP_N, 'ssim')['image_id'].values
    worst_ssim_idx = df.nsmallest(TOP_N, 'ssim')['image_id'].values
    
    # Create comparison grids
    create_highlight_grid(metrics, images, best_psnr_idx, os.path.join(highlight_dir, 'best_psnr.png'), 'Best PSNR Images')
    create_highlight_grid(metrics, images, worst_psnr_idx, os.path.join(highlight_dir, 'worst_psnr.png'), 'Worst PSNR Images')
    create_highlight_grid(metrics, images, best_ssim_idx, os.path.join(highlight_dir, 'best_ssim.png'), 'Best SSIM Images')
    create_highlight_grid(metrics, images, worst_ssim_idx, os.path.join(highlight_dir, 'worst_ssim.png'), 'Worst SSIM Images')
    
    print(f"Saved highlight images to {highlight_dir}")

def create_highlight_grid(metrics, images, indices, output_path, title):
    """Create a grid of images with grayscale, colorized, and original side by side"""
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(15, n*4))
    
    if n == 1:
        axes = [axes]  # Handle the case with just one image
    
    for i, idx in enumerate(indices):
        # Get original, grayscale and comparison images
        gray_img = images['grayscales'][idx]
        original_img = images['originals'][idx]
        comparison_img = images['comparisons'][idx]
        colorized_img = extract_colorization_from_comparison(comparison_img, original_img)
        
        # Display images
        axes[i][0].imshow(gray_img, cmap='gray')
        axes[i][0].set_title('Grayscale Input')
        axes[i][0].axis('off')
        
        axes[i][1].imshow(colorized_img)
        axes[i][1].set_title('Colorized Output')
        axes[i][1].axis('off')
        
        axes[i][2].imshow(original_img)
        axes[i][2].set_title('Original')
        axes[i][2].axis('off')
        
        # Add metrics as text on the right
        filename = os.path.basename(images['comparison_paths'][idx])
        try:
            result = next(r for r in metrics['individual_results'] if r['image_id'] == idx)
            metrics_text = f"File: {filename}\nPSNR: {result['psnr']:.2f}\nSSIM: {result['ssim']:.4f}"
            
            # Add other metrics if available
            if 'lpips' in result and not np.isnan(result['lpips']):
                metrics_text += f"\nLPIPS: {result['lpips']:.4f}"
            if 'deltaE' in result and not np.isnan(result['deltaE']):
                metrics_text += f"\nΔE: {result['deltaE']:.2f}"
                
            axes[i][2].text(1.05, 0.5, 
                            metrics_text,
                            transform=axes[i][2].transAxes, 
                            fontsize=9,
                            verticalalignment='center')
        except (StopIteration, KeyError) as e:
            print(f"Warning: Could not find metrics for image {idx}")
            axes[i][2].text(1.05, 0.5, 
                            f"File: {filename}\nMetrics unavailable",
                            transform=axes[i][2].transAxes, 
                            fontsize=9,
                            verticalalignment='center')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)  # Adjust to make room for suptitle
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def create_summary_dataframe(metrics, output_path):
    """Create and save a summary DataFrame with all metrics"""
    df = pd.DataFrame(metrics['individual_results'])
    
    # Add best/worst tags
    top_n_psnr = df.nlargest(TOP_N, 'psnr')['image_id'].values
    bottom_n_psnr = df.nsmallest(TOP_N, 'psnr')['image_id'].values
    top_n_ssim = df.nlargest(TOP_N, 'ssim')['image_id'].values
    bottom_n_ssim = df.nsmallest(TOP_N, 'ssim')['image_id'].values
    
    df['best_psnr'] = df['image_id'].isin(top_n_psnr)
    df['worst_psnr'] = df['image_id'].isin(bottom_n_psnr)
    df['best_ssim'] = df['image_id'].isin(top_n_ssim)
    df['worst_ssim'] = df['image_id'].isin(bottom_n_ssim)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved summary DataFrame to {output_path}")
    return df

def calculate_fid(real_dir, fake_dir):
    """Calculate FID score using torch-fidelity if directories exist"""
    try:
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            print(f"FID calculation skipped. Create {real_dir}/ and {fake_dir}/ directories with images to enable it.")
            return None
        
        # Import only when needed
        from torch_fidelity import calculate_metrics as calc_fid_metrics
        
        print(f"Calculating FID between {real_dir} and {fake_dir}...")
        metrics = calc_fid_metrics(
            input1=real_dir,
            input2=fake_dir,
            cuda=True,
            fid=True,
            verbose=False
        )
        
        print(f"FID Score: {metrics['frechet_inception_distance']:.4f}")
        return metrics['frechet_inception_distance']
    except ImportError:
        print("torch-fidelity not available. Install it with 'pip install torch-fidelity'.")
        return None
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return None

def main():
    # Find the latest results directory
    result_dir = find_latest_run()
    
    # Create plots directory
    plots_dir = PLOTS_DIR
    ensure_dir(plots_dir)
    
    # Load images
    print("Loading images...")
    images = load_images(result_dir)
    
    # Load or compute metrics
    print("Analyzing metrics...")
    metrics = load_or_compute_metrics(images, result_dir)
    
    # Print metrics summary
    print("\n===== GAN Colorization Results =====")
    print(f"PSNR: {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
    
    if 'lpips' in metrics:
        print(f"LPIPS: {metrics['lpips']:.4f} ± {metrics['lpips_std']:.4f}")
    
    if 'deltaE' in metrics:
        print(f"ΔE (CIEDE2000): {metrics['deltaE']:.2f} ± {metrics['deltaE_std']:.2f}")
    
    if 'hist_corr' in metrics:
        print(f"Histogram Correlation: {metrics['hist_corr']:.4f} ± {metrics['hist_corr_std']:.4f}")
    
    # Create visualizations
    print("\nGenerating visualization plots...")
    create_metric_plots(metrics, plots_dir)
    
    # Highlight best and worst images
    print("Creating highlight comparisons...")
    highlight_best_worst(metrics, images, plots_dir)
    
    # Create summary DataFrame
    summary_path = os.path.join(result_dir, SUMMARY_FILE)
    df = create_summary_dataframe(metrics, summary_path)
    
    # FID calculation (optional - commented out by default)
    """
    # Uncomment the following lines to calculate FID
    real_dir = "real"
    fake_dir = "fake"
    fid_score = calculate_fid(real_dir, fake_dir)
    if fid_score is not None:
        print(f"\nFID Score: {fid_score:.4f}")
    """
    print(f"\nTo calculate FID score:")
    print("1. Create 'real/' and 'fake/' directories with original and generated images")
    print("2. Uncomment the FID calculation section in this script")
    
    # Print best and worst image details
    best_psnr = df.nlargest(TOP_N, 'psnr')
    worst_psnr = df.nsmallest(TOP_N, 'psnr')
    
    print(f"\nTop {TOP_N} best images by PSNR:")
    for i, (_, row) in enumerate(best_psnr.iterrows()):
        print(f"  {i+1}. {row['filename']} - PSNR: {row['psnr']:.2f}, SSIM: {row['ssim']:.4f}")
    
    print(f"\nTop {TOP_N} worst images by PSNR:")
    for i, (_, row) in enumerate(worst_psnr.iterrows()):
        print(f"  {i+1}. {row['filename']} - PSNR: {row['psnr']:.2f}, SSIM: {row['ssim']:.4f}")
    
    print("\nEvaluation completed successfully!")
    print(f"- Plots saved to: {plots_dir}")
    print(f"- Summary data saved to: {summary_path}")

if __name__ == "__main__":
    main() 