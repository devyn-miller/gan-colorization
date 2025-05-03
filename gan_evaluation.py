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
import torch
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import cv2
import re

# Optional imports for additional metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not available - install with 'pip install lpips'")

try:
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color
    from colormath.color_diff import delta_e_cie2000
    DELTA_E_AVAILABLE = True
except ImportError:
    DELTA_E_AVAILABLE = False
    print("Delta E calculation not available - install with 'pip install colormath'")

# Configuration
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
HIGHLIGHTS_DIR = os.path.join(PLOTS_DIR, "highlights")
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

def extract_image_number(filename):
    """Extract image number from filename (e.g., 'grayscale_05.png' -> 5)"""
    match = re.search(r'_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return -1

def load_images(result_dir):
    """Load all result images from the specified directory"""
    # Find all image files
    grayscale_images = sorted(glob.glob(os.path.join(result_dir, "grayscale_*.png")))
    original_images = sorted(glob.glob(os.path.join(result_dir, "original_*.png")))
    predicted_images = sorted(glob.glob(os.path.join(result_dir, "comparison_*.png")))
    
    # Group by image number and create a dictionary
    images_by_number = {}
    
    for img_path in grayscale_images:
        img_num = extract_image_number(img_path)
        if img_num not in images_by_number:
            images_by_number[img_num] = {}
        images_by_number[img_num]['grayscale_path'] = img_path
    
    for img_path in original_images:
        img_num = extract_image_number(img_path)
        if img_num not in images_by_number:
            images_by_number[img_num] = {}
        images_by_number[img_num]['original_path'] = img_path
    
    for img_path in predicted_images:
        img_num = extract_image_number(img_path)
        if img_num not in images_by_number:
            images_by_number[img_num] = {}
        images_by_number[img_num]['predicted_path'] = img_path
    
    # Filter out incomplete entries and load images
    complete_images = []
    for img_num, paths in images_by_number.items():
        if all(k in paths for k in ['grayscale_path', 'original_path', 'predicted_path']):
            grayscale = imread(paths['grayscale_path'])
            original = imread(paths['original_path'])
            predicted_full = imread(paths['predicted_path'])
            
            # Extract the predicted image from the comparison image
            # Assuming the middle part of the 3-part comparison is the predicted color
            h, w, _ = predicted_full.shape
            predicted = predicted_full[:, w//3:2*w//3, :]
            
            # Ensure dimensions match
            if grayscale.shape[:2] != original.shape[:2]:
                # Resize grayscale to match original size
                grayscale = cv2.resize(grayscale, (original.shape[1], original.shape[0]))
            
            if predicted.shape[:2] != original.shape[:2]:
                # Resize predicted to match original size
                predicted = cv2.resize(predicted, (original.shape[1], original.shape[0]))
            
            complete_images.append({
                'img_num': img_num,
                'grayscale_path': paths['grayscale_path'],
                'original_path': paths['original_path'],
                'predicted_path': paths['predicted_path'],
                'grayscale': grayscale,
                'original': original,
                'predicted': predicted,
                'predicted_full': predicted_full
            })
    
    print(f"Loaded {len(complete_images)} complete image sets")
    return complete_images

def compute_psnr_ssim(img1, img2):
    """Compute PSNR and SSIM metrics between two images"""
    try:
        # Calculate PSNR
        psnr_value = psnr(img1, img2)
        
        # Calculate SSIM
        ssim_value = ssim(img1, img2, channel_axis=2, data_range=1.0)
        
        return psnr_value, ssim_value
    except Exception as e:
        print(f"Error calculating PSNR/SSIM: {e}")
        return float('nan'), float('nan')

def compute_delta_e(img1, img2, sample_size=100):
    """Compute Delta E color difference between two images using random sampling"""
    if not DELTA_E_AVAILABLE:
        return float('nan')
    
    try:
        h, w, _ = img1.shape
        delta_e_samples = []
        
        for _ in range(sample_size):  # Sample random pixels
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            rgb1 = sRGBColor(img1[y, x, 0], img1[y, x, 1], img1[y, x, 2], is_upscaled=True)
            rgb2 = sRGBColor(img2[y, x, 0], img2[y, x, 1], img2[y, x, 2], is_upscaled=True)
            lab1 = convert_color(rgb1, LabColor)
            lab2 = convert_color(rgb2, LabColor)
            delta_e = delta_e_cie2000(lab1, lab2)
            delta_e_samples.append(delta_e)
            
        return np.mean(delta_e_samples)
    except Exception as e:
        print(f"Error calculating Delta E: {e}")
        return float('nan')

def compute_histogram_correlation(img1, img2):
    """Compute histogram correlation between two RGB images"""
    try:
        hist_corr = 0
        for ch in range(3):  # For each RGB channel
            hist1 = cv2.calcHist([img1], [ch], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [ch], None, [256], [0, 256])
            hist_corr += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return hist_corr / 3.0  # Average across channels
    except Exception as e:
        print(f"Error calculating histogram correlation: {e}")
        return float('nan')

def compute_lpips(img1, img2):
    """Compute LPIPS perceptual similarity between two images"""
    if not LPIPS_AVAILABLE:
        return float('nan')
    
    try:
        # Initialize LPIPS model
        lpips_model = lpips.LPIPS(net='alex')
        
        # Convert images to PyTorch tensors and normalize to [-1, 1]
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_value = lpips_model(img1_tensor, img2_tensor).item()
        
        return lpips_value
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
        return float('nan')

def compute_metrics(images):
    """Compute all metrics for each image set"""
    metrics = []
    
    for img_set in images:
        # Extract images
        original = img_set['original']
        predicted = img_set['predicted']
        
        # Compute basic metrics (PSNR, SSIM)
        psnr_value, ssim_value = compute_psnr_ssim(original, predicted)
        
        # Compute optional metrics
        delta_e = compute_delta_e(original, predicted) if DELTA_E_AVAILABLE else float('nan')
        hist_corr = compute_histogram_correlation(original, predicted)
        lpips_value = compute_lpips(original, predicted) if LPIPS_AVAILABLE else float('nan')
        
        # Add results to metrics list
        metrics.append({
            'img_num': img_set['img_num'],
            'grayscale_path': img_set['grayscale_path'],
            'original_path': img_set['original_path'],
            'predicted_path': img_set['predicted_path'],
            'psnr': psnr_value,
            'ssim': ssim_value,
            'delta_e': delta_e,
            'hist_corr': hist_corr,
            'lpips': lpips_value
        })
    
    # Calculate summary statistics
    psnr_values = [m['psnr'] for m in metrics if not np.isnan(m['psnr'])]
    ssim_values = [m['ssim'] for m in metrics if not np.isnan(m['ssim'])]
    
    summary = {
        'mean_psnr': np.mean(psnr_values),
        'std_psnr': np.std(psnr_values),
        'min_psnr': np.min(psnr_values),
        'max_psnr': np.max(psnr_values),
        'mean_ssim': np.mean(ssim_values),
        'std_ssim': np.std(ssim_values),
        'min_ssim': np.min(ssim_values),
        'max_ssim': np.max(ssim_values),
    }
    
    # Add optional metrics to summary if available
    if DELTA_E_AVAILABLE:
        delta_e_values = [m['delta_e'] for m in metrics if not np.isnan(m['delta_e'])]
        if delta_e_values:
            summary['mean_delta_e'] = np.mean(delta_e_values)
            summary['std_delta_e'] = np.std(delta_e_values)
    
    if LPIPS_AVAILABLE:
        lpips_values = [m['lpips'] for m in metrics if not np.isnan(m['lpips'])]
        if lpips_values:
            summary['mean_lpips'] = np.mean(lpips_values)
            summary['std_lpips'] = np.std(lpips_values)
    
    hist_corr_values = [m['hist_corr'] for m in metrics if not np.isnan(m['hist_corr'])]
    if hist_corr_values:
        summary['mean_hist_corr'] = np.mean(hist_corr_values)
        summary['std_hist_corr'] = np.std(hist_corr_values)
    
    return metrics, summary

def save_metrics_to_json(metrics, summary, output_file):
    """Save metrics and summary to a JSON file"""
    # Create a serializable version of the metrics
    serializable_metrics = []
    for m in metrics:
        serializable_m = {}
        for k, v in m.items():
            if isinstance(v, np.float32) or isinstance(v, np.float64):
                serializable_m[k] = float(v)
            elif isinstance(v, np.int32) or isinstance(v, np.int64):
                serializable_m[k] = int(v)
            else:
                serializable_m[k] = v
        serializable_metrics.append(serializable_m)
    
    # Create serializable summary
    serializable_summary = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                            for k, v in summary.items()}
    
    # Combine data
    data = {
        'metrics': serializable_metrics,
        'summary': serializable_summary,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Metrics saved to {output_file}")

def create_summary_dataframe(metrics, output_file):
    """Create and save a pandas DataFrame summary of all metrics"""
    # Extract relevant data
    data = []
    for m in metrics:
        data.append({
            'Image': m['img_num'],
            'PSNR': m['psnr'],
            'SSIM': m['ssim'],
            'DeltaE': m['delta_e'] if not np.isnan(m['delta_e']) else None,
            'HistCorr': m['hist_corr'] if not np.isnan(m['hist_corr']) else None,
            'LPIPS': m['lpips'] if not np.isnan(m['lpips']) else None,
            'GrayscalePath': os.path.basename(m['grayscale_path']),
            'OriginalPath': os.path.basename(m['original_path']),
            'PredictedPath': os.path.basename(m['predicted_path'])
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")
    
    return df

def create_metric_histograms(metrics, output_dir):
    """Create histograms for PSNR and SSIM distributions"""
    ensure_dir(output_dir)
    
    # Extract values
    psnr_values = [m['psnr'] for m in metrics if not np.isnan(m['psnr'])]
    ssim_values = [m['ssim'] for m in metrics if not np.isnan(m['ssim'])]
    
    # PSNR histogram
    plt.figure(figsize=(10, 6))
    plt.hist(psnr_values, bins=15, alpha=0.7, color='blue')
    plt.axvline(np.mean(psnr_values), color='red', linestyle='dashed', linewidth=2)
    plt.title(f'PSNR Distribution (Mean: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f})')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'psnr_distribution.png'), dpi=100, bbox_inches='tight')
    plt.close()
    
    # SSIM histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ssim_values, bins=15, alpha=0.7, color='green')
    plt.axvline(np.mean(ssim_values), color='red', linestyle='dashed', linewidth=2)
    plt.title(f'SSIM Distribution (Mean: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f})')
    plt.xlabel('SSIM')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'ssim_distribution.png'), dpi=100, bbox_inches='tight')
    plt.close()
    
    # Additional histograms for optional metrics
    if DELTA_E_AVAILABLE:
        delta_e_values = [m['delta_e'] for m in metrics if not np.isnan(m['delta_e'])]
        if delta_e_values:
            plt.figure(figsize=(10, 6))
            plt.hist(delta_e_values, bins=15, alpha=0.7, color='orange')
            plt.axvline(np.mean(delta_e_values), color='red', linestyle='dashed', linewidth=2)
            plt.title(f'Delta E Distribution (Mean: {np.mean(delta_e_values):.2f} ± {np.std(delta_e_values):.2f})')
            plt.xlabel('Delta E')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, 'delta_e_distribution.png'), dpi=100, bbox_inches='tight')
            plt.close()
    
    if LPIPS_AVAILABLE:
        lpips_values = [m['lpips'] for m in metrics if not np.isnan(m['lpips'])]
        if lpips_values:
            plt.figure(figsize=(10, 6))
            plt.hist(lpips_values, bins=15, alpha=0.7, color='purple')
            plt.axvline(np.mean(lpips_values), color='red', linestyle='dashed', linewidth=2)
            plt.title(f'LPIPS Distribution (Mean: {np.mean(lpips_values):.4f} ± {np.std(lpips_values):.4f})')
            plt.xlabel('LPIPS')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, 'lpips_distribution.png'), dpi=100, bbox_inches='tight')
            plt.close()

def create_metric_correlation_plot(metrics, output_dir):
    """Create scatter plot to visualize correlation between PSNR and SSIM"""
    ensure_dir(output_dir)
    
    # Extract valid pairs of metrics
    valid_indices = [i for i, m in enumerate(metrics) 
                    if not np.isnan(m['psnr']) and not np.isnan(m['ssim'])]
    psnr_values = [metrics[i]['psnr'] for i in valid_indices]
    ssim_values = [metrics[i]['ssim'] for i in valid_indices]
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(psnr_values, ssim_values, alpha=0.7, c='blue')
    
    # Add trend line
    z = np.polyfit(psnr_values, ssim_values, 1)
    p = np.poly1d(z)
    plt.plot(psnr_values, p(psnr_values), "r--", alpha=0.7)
    
    # Calculate correlation coefficient
    corr = np.corrcoef(psnr_values, ssim_values)[0, 1]
    
    plt.title(f'PSNR vs SSIM (Correlation: {corr:.4f})')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('SSIM')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'psnr_vs_ssim.png'), dpi=100, bbox_inches='tight')
    plt.close()

def create_side_by_side_comparison(img_set, title, output_path):
    """Create a side-by-side comparison of grayscale, predicted, and original images"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot grayscale
    axes[0].imshow(img_set['grayscale'], cmap='gray')
    axes[0].set_title('Grayscale Input')
    axes[0].axis('off')
    
    # Plot predicted
    axes[1].imshow(img_set['predicted'])
    axes[1].set_title('GAN Colorization')
    axes[1].axis('off')
    
    # Plot original
    axes[2].imshow(img_set['original'])
    axes[2].set_title('Original Color')
    axes[2].axis('off')
    
    # Add main title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def highlight_best_worst(images, metrics, output_dir):
    """Create visualizations for the best and worst performing images"""
    ensure_dir(output_dir)
    
    # Find indices of best and worst images by PSNR
    valid_indices = [i for i, m in enumerate(metrics) if not np.isnan(m['psnr'])]
    sorted_by_psnr = sorted(valid_indices, key=lambda i: metrics[i]['psnr'], reverse=True)
    
    best_psnr_indices = sorted_by_psnr[:TOP_N]
    worst_psnr_indices = sorted_by_psnr[-TOP_N:]
    
    # Find indices of best and worst images by SSIM
    valid_indices = [i for i, m in enumerate(metrics) if not np.isnan(m['ssim'])]
    sorted_by_ssim = sorted(valid_indices, key=lambda i: metrics[i]['ssim'], reverse=True)
    
    best_ssim_indices = sorted_by_ssim[:TOP_N]
    worst_ssim_indices = sorted_by_ssim[-TOP_N:]
    
    # Create highlight visualizations
    for i, idx in enumerate(best_psnr_indices):
        img_set = next(img for img in images if img['img_num'] == metrics[idx]['img_num'])
        title = f"Best PSNR #{i+1}: PSNR={metrics[idx]['psnr']:.2f}, SSIM={metrics[idx]['ssim']:.4f}"
        output_path = os.path.join(output_dir, f"best_psnr_{i+1}.png")
        create_side_by_side_comparison(img_set, title, output_path)
    
    for i, idx in enumerate(worst_psnr_indices):
        img_set = next(img for img in images if img['img_num'] == metrics[idx]['img_num'])
        title = f"Worst PSNR #{i+1}: PSNR={metrics[idx]['psnr']:.2f}, SSIM={metrics[idx]['ssim']:.4f}"
        output_path = os.path.join(output_dir, f"worst_psnr_{i+1}.png")
        create_side_by_side_comparison(img_set, title, output_path)
    
    for i, idx in enumerate(best_ssim_indices):
        img_set = next(img for img in images if img['img_num'] == metrics[idx]['img_num'])
        title = f"Best SSIM #{i+1}: SSIM={metrics[idx]['ssim']:.4f}, PSNR={metrics[idx]['psnr']:.2f}"
        output_path = os.path.join(output_dir, f"best_ssim_{i+1}.png")
        create_side_by_side_comparison(img_set, title, output_path)
    
    for i, idx in enumerate(worst_ssim_indices):
        img_set = next(img for img in images if img['img_num'] == metrics[idx]['img_num'])
        title = f"Worst SSIM #{i+1}: SSIM={metrics[idx]['ssim']:.4f}, PSNR={metrics[idx]['psnr']:.2f}"
        output_path = os.path.join(output_dir, f"worst_ssim_{i+1}.png")
        create_side_by_side_comparison(img_set, title, output_path)

    # Create combined grid visualizations
    create_grid_visualization([images[i] for i in best_psnr_indices], 
                           [metrics[i] for i in best_psnr_indices],
                           os.path.join(output_dir, "best_psnr.png"),
                           "Best PSNR Images")
    
    create_grid_visualization([images[i] for i in worst_psnr_indices],
                           [metrics[i] for i in worst_psnr_indices],
                           os.path.join(output_dir, "worst_psnr.png"),
                           "Worst PSNR Images")
    
    create_grid_visualization([images[i] for i in best_ssim_indices],
                           [metrics[i] for i in best_ssim_indices],
                           os.path.join(output_dir, "best_ssim.png"),
                           "Best SSIM Images")
    
    create_grid_visualization([images[i] for i in worst_ssim_indices],
                           [metrics[i] for i in worst_ssim_indices],
                           os.path.join(output_dir, "worst_ssim.png"),
                           "Worst SSIM Images")

def create_grid_visualization(img_sets, metrics_list, output_path, title):
    """Create a grid visualization of multiple image sets"""
    n_images = len(img_sets)
    if n_images == 0:
        return
    
    # Determine grid dimensions (trying to be roughly square)
    n_cols = min(3, n_images)  # At most 3 columns
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    
    # In case of a single row, reshape axes
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each image set
    for i, (img_set, metrics) in enumerate(zip(img_sets, metrics_list)):
        row = i // n_cols
        col = i % n_cols
        
        # Get the axes for this image set
        ax_gray = axes[row, col * 3]
        ax_pred = axes[row, col * 3 + 1]
        ax_orig = axes[row, col * 3 + 2]
        
        # Plot grayscale
        ax_gray.imshow(img_set['grayscale'], cmap='gray')
        ax_gray.set_title('Grayscale')
        ax_gray.axis('off')
        
        # Plot predicted
        ax_pred.imshow(img_set['predicted'])
        ax_pred.set_title('Predicted')
        ax_pred.axis('off')
        
        # Plot original
        ax_orig.imshow(img_set['original'])
        metric_text = f"PSNR: {metrics['psnr']:.2f}\nSSIM: {metrics['ssim']:.4f}"
        ax_orig.set_title(f'Original\n{metric_text}')
        ax_orig.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        for j in range(3):
            axes[row, col * 3 + j].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def generate_report(summary, metrics, run_dir, output_file):
    """Generate a detailed text report of the evaluation results"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GAN COLORIZATION EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results directory: {run_dir}\n")
        f.write(f"Number of evaluated images: {len(metrics)}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n\n")
        
        f.write(f"PSNR (Peak Signal-to-Noise Ratio):\n")
        f.write(f"  Mean: {summary['mean_psnr']:.2f} dB\n")
        f.write(f"  Standard Deviation: {summary['std_psnr']:.2f} dB\n")
        f.write(f"  Range: {summary['min_psnr']:.2f} - {summary['max_psnr']:.2f} dB\n\n")
        
        f.write(f"SSIM (Structural Similarity Index):\n")
        f.write(f"  Mean: {summary['mean_ssim']:.4f}\n")
        f.write(f"  Standard Deviation: {summary['std_ssim']:.4f}\n")
        f.write(f"  Range: {summary['min_ssim']:.4f} - {summary['max_ssim']:.4f}\n\n")
        
        # Add optional metrics if available
        if 'mean_delta_e' in summary:
            f.write(f"Delta E (Color Difference):\n")
            f.write(f"  Mean: {summary['mean_delta_e']:.2f}\n")
            f.write(f"  Standard Deviation: {summary['std_delta_e']:.2f}\n\n")
        
        if 'mean_lpips' in summary:
            f.write(f"LPIPS (Learned Perceptual Image Patch Similarity):\n")
            f.write(f"  Mean: {summary['mean_lpips']:.4f}\n")
            f.write(f"  Standard Deviation: {summary['std_lpips']:.4f}\n\n")
        
        if 'mean_hist_corr' in summary:
            f.write(f"Histogram Correlation:\n")
            f.write(f"  Mean: {summary['mean_hist_corr']:.4f}\n")
            f.write(f"  Standard Deviation: {summary['std_hist_corr']:.4f}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("BEST PERFORMING IMAGES\n")
        f.write("-" * 40 + "\n\n")
        
        # Sort by PSNR
        sorted_by_psnr = sorted(metrics, key=lambda m: m['psnr'] if not np.isnan(m['psnr']) else -float('inf'), reverse=True)
        
        f.write("Top 5 by PSNR:\n")
        for i, m in enumerate(sorted_by_psnr[:5]):
            f.write(f"  {i+1}. Image {m['img_num']}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}\n")
        
        f.write("\nTop 5 by SSIM:\n")
        sorted_by_ssim = sorted(metrics, key=lambda m: m['ssim'] if not np.isnan(m['ssim']) else -float('inf'), reverse=True)
        for i, m in enumerate(sorted_by_ssim[:5]):
            f.write(f"  {i+1}. Image {m['img_num']}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.2f}\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write("WORST PERFORMING IMAGES\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("Bottom 5 by PSNR:\n")
        for i, m in enumerate(sorted_by_psnr[-5:]):
            f.write(f"  {i+1}. Image {m['img_num']}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}\n")
        
        f.write("\nBottom 5 by SSIM:\n")
        for i, m in enumerate(sorted_by_ssim[-5:]):
            f.write(f"  {i+1}. Image {m['img_num']}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.2f}\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write("VISUALIZATIONS GENERATED\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("1. Metric Distributions:\n")
        f.write("   - PSNR distribution histogram\n")
        f.write("   - SSIM distribution histogram\n")
        
        if DELTA_E_AVAILABLE:
            f.write("   - Delta E distribution histogram\n")
        
        if LPIPS_AVAILABLE:
            f.write("   - LPIPS distribution histogram\n")
        
        f.write("\n2. Metric Correlations:\n")
        f.write("   - PSNR vs SSIM scatter plot\n")
        
        f.write("\n3. Best/Worst Image Highlights:\n")
        f.write("   - Best/worst by PSNR - individual comparisons\n")
        f.write("   - Best/worst by SSIM - individual comparisons\n")
        f.write("   - Combined grid visualizations\n")
        
        f.write("\n\nFull evaluation data saved to CSV and JSON files.\n")
        f.write("End of report.\n")
    
    print(f"Report generated and saved to {output_file}")

def main():
    """Main function to execute the evaluation pipeline"""
    print("Starting GAN colorization evaluation...")
    
    # Ensure output directories exist
    ensure_dir(RESULTS_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(HIGHLIGHTS_DIR)
    
    # Find latest run directory
    try:
        run_dir = find_latest_run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load images from the run directory
    images = load_images(run_dir)
    if not images:
        print("No valid images found in the results directory.")
        return
    
    # Compute metrics
    print("Computing image quality metrics...")
    metrics, summary = compute_metrics(images)
    
    # Save metrics to JSON
    metrics_file = os.path.join(run_dir, METRICS_FILE)
    save_metrics_to_json(metrics, summary, metrics_file)
    
    # Create summary DataFrame
    summary_file = os.path.join(run_dir, SUMMARY_FILE)
    create_summary_dataframe(metrics, summary_file)
    
    # Create visualizations
    print("Generating visualizations...")
    create_metric_histograms(metrics, PLOTS_DIR)
    create_metric_correlation_plot(metrics, PLOTS_DIR)
    
    # Highlight best and worst images
    print("Creating best/worst image highlights...")
    highlight_best_worst(images, metrics, HIGHLIGHTS_DIR)
    
    # Generate report
    report_file = os.path.join(run_dir, "evaluation_report.txt")
    generate_report(summary, metrics, run_dir, report_file)
    
    print("\nEvaluation complete!")
    print(f"Summary results:")
    print(f"  PSNR: {summary['mean_psnr']:.2f} ± {summary['std_psnr']:.2f} dB")
    print(f"  SSIM: {summary['mean_ssim']:.4f} ± {summary['std_ssim']:.4f}")
    print(f"\nResults saved to {run_dir}")
    print(f"Visualizations saved to {PLOTS_DIR} and {HIGHLIGHTS_DIR}")

if __name__ == "__main__":
    main() 