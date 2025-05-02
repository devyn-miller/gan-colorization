import numpy as np
from PIL import Image
import os
import glob
from skimage.color import rgb2lab
import torch
from torch.utils.data import DataLoader, Dataset

def load_data(stack, data_dir='data/images', img_size=256, batch_size=4):
    """
    Load and prepare image data for GAN colorization training
    
    Args:
        stack (Stack): The Stack object to update with data generators
        data_dir (str): Directory containing training images
        img_size (int): Size to resize images to (square)
        batch_size (int): Batch size for generators
        
    Returns:
        Stack: Updated Stack object with data generators
    """
    print(f"Loading data from {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        # If not, create a sample dataset with random data for testing
        print(f"Data directory {data_dir} not found, creating synthetic data for testing")
        return create_synthetic_data(stack, img_size=img_size, batch_size=batch_size)
    
    # Get list of image files
    image_files = glob.glob(f"{data_dir}/**/*.jpg", recursive=True)
    image_files.extend(glob.glob(f"{data_dir}/**/*.png", recursive=True))
    
    if len(image_files) == 0:
        print(f"No image files found in {data_dir}, creating synthetic data")
        return create_synthetic_data(stack, img_size=img_size, batch_size=batch_size)
    
    print(f"Found {len(image_files)} images")
    
    # Split data into train, validation, and test sets
    np.random.shuffle(image_files)
    train_files = image_files[:int(0.8 * len(image_files))]
    val_files = image_files[int(0.8 * len(image_files)):int(0.9 * len(image_files))]
    test_files = image_files[int(0.9 * len(image_files)):]
    
    # Create data generators
    def create_generator(file_list, is_train=False):
        def generator():
            # Shuffle on each epoch if training
            if is_train:
                np.random.shuffle(file_list)
            
            # Process in batches
            batch = []
            for img_path in file_list:
                try:
                    # Load and resize image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((img_size, img_size))
                    img_array = np.array(img) / 255.0  # Normalize to [0,1]
                    
                    # Add to batch
                    batch.append(img_array)
                    
                    # Yield batch when it reaches the batch size
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Yield remaining samples
            if batch:
                yield batch
        
        return generator()
    
    # Update stack with generators
    stack.update_datasets(
        train_generator=create_generator(train_files, is_train=True),
        val_generator=create_generator(val_files),
        test_generator=create_generator(test_files)
    )
    
    # Update image dimensions
    stack.update_dimensions(img_size, img_size)
    
    return stack

def create_synthetic_data(stack, img_size=256, batch_size=4, num_samples=100):
    """
    Create synthetic data for testing when real data is not available
    
    Args:
        stack (Stack): The Stack object to update
        img_size (int): Image size
        batch_size (int): Batch size
        num_samples (int): Number of synthetic samples to generate
        
    Returns:
        Stack: Updated Stack object
    """
    print("Creating synthetic data for testing")
    
    # Create synthetic image data
    synthetic_images = []
    for _ in range(num_samples):
        # Create a random RGB image
        img = np.random.rand(img_size, img_size, 3)
        synthetic_images.append(img)
    
    # Split into train, val, test
    train_images = synthetic_images[:int(0.8 * num_samples)]
    val_images = synthetic_images[int(0.8 * num_samples):int(0.9 * num_samples)]
    test_images = synthetic_images[int(0.9 * num_samples):]
    
    # Create generators
    def create_generator(images):
        def generator():
            # Process in batches
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                if len(batch) == batch_size:  # Only yield complete batches
                    yield batch
        
        return generator()
    
    # Update stack with generators
    stack.update_datasets(
        train_generator=create_generator(train_images),
        val_generator=create_generator(val_images),
        test_generator=create_generator(test_images)
    )
    
    # Update image dimensions
    stack.update_dimensions(img_size, img_size)
    
    return stack 