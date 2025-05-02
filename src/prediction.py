import tensorflow as tf
import torch
import numpy as np
from skimage.color import rgb2lab, lab2rgb

from objects.stack import Stack
from objects.result import Result
from preprocessing import preprocess
from gan_utils import lab_to_rgb


class Predictor:
    def __init__(self, model_path, model_type='gan'):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path (str): Path to the trained model weights
            model_type (str): Type of model ('gan' or 'autoencoder')
        """
        self.model_type = model_type
        self.result = Result()

        if model_type == 'autoencoder':
            # Load TensorFlow model
            self.model = tf.keras.models.load_model(model_path)
        elif model_type == 'gan':
            # Load PyTorch generator model
            self.stack = Stack()
            self.model = self.stack.load_gan_generator(
                model_path=model_path,
                input_channels=1,
                output_channels=2,
                model_type='resnet_unet'  # or 'unet'
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def predict(self, input_data, return_rgb=True):
        """
        Make predictions using the loaded model and input data.

        Args:
            input_data: Input grayscale image(s)
            return_rgb (bool): Whether to return RGB images or LAB components

        Returns:
            np.ndarray: Colorized images or LAB components
        """
        if self.model_type == 'autoencoder':
            # For autoencoder model (TensorFlow)
            processed_data = preprocess(input_data)
            predictions = self.model.predict(processed_data)

            if hasattr(self.result, 'save_predictions'):
                self.result.save_predictions(predictions)
            else:
                raise AttributeError("Result class does not have a 'save_predictions' method")

            return predictions

        # For GAN model (PyTorch)
        if isinstance(input_data, np.ndarray):
            if len(input_data.shape) == 2:
                # Single grayscale image
                L = input_data[np.newaxis, np.newaxis, :, :] / 50.0 - 1.0
            elif len(input_data.shape) == 3 and input_data.shape[2] == 1:
                # Batch of grayscale images with channel dimension
                L = input_data.transpose(0, 3, 1, 2) / 50.0 - 1.0
            elif len(input_data.shape) == 3:
                # Single RGB image
                lab = rgb2lab(input_data).astype('float32')
                L = lab[:, :, 0][np.newaxis, np.newaxis, :, :] / 50.0 - 1.0
            elif len(input_data.shape) == 4 and input_data.shape[3] == 3:
                # Batch of RGB images
                L_list = []
                for img in input_data:
                    lab = rgb2lab(img).astype('float32')
                    L_list.append(lab[:, :, 0])
                L = np.stack(L_list)[:, np.newaxis, :, :] / 50.0 - 1.0
            else:
                raise ValueError(f"Unsupported input shape: {input_data.shape}")

            L_tensor = torch.from_numpy(L).float().to(self.device)
        else:
            raise ValueError("Input data should be a numpy array")

        # Generate predictions using GAN
        with torch.no_grad():
            ab_pred = self.model(L_tensor)

        ab_pred_np = ab_pred.cpu().numpy()
        L_np = (L_tensor.cpu().numpy() + 1.0) * 50.0  # Scale back to [0, 100]

        if return_rgb:
            rgb_images = []
            for i in range(L_np.shape[0]):
                L_img = L_np[i, 0, :, :]
                ab_img = ab_pred_np[i] * 110.0  # Scale back to [-110, 110]
                lab_img = np.concatenate([L_img[:, :, np.newaxis], ab_img.transpose(1, 2, 0)], axis=2)
                rgb_img = lab2rgb(lab_img)
                rgb_images.append(rgb_img)
            rgb_images = np.stack(rgb_images)

            if hasattr(self.result, 'save_predictions'):
                self.result.save_predictions(rgb_images)

            return rgb_images
        else:
            return {
                'L': L_np,
                'ab_pred': ab_pred_np
            }
