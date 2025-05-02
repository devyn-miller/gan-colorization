import tensorflow.keras.models
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, lab2rgb

from objects.result import Result
from objects.stack import Stack
from preprocessing import preprocess 
from gan_utils import calculate_metrics


class Validator:
    def __init__(self, model_path, model_type='gan'):
        """
        Initialize the validator with a trained model.
        
        Args:
            model_path (str): Path to the trained model
            model_type (str): Type of model ('gan' or 'autoencoder')
        """
        self.model_type = model_type
        self.result = Result()

        if model_type == 'autoencoder':
            self.model = tensorflow.keras.models.load_model(model_path)
        elif model_type == 'gan':
            self.stack = Stack()
            self.model = self.stack.load_gan_generator(
                model_path=model_path,
                input_channels=1, 
                output_channels=2, 
                model_type='resnet_unet'  # or 'unet'
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def validate(self, validation_data, ground_truth=None):
        """
        Evaluates the model on the validation data.
        
        Args:
            validation_data: RGB or grayscale images (NumPy array)
            ground_truth: Ground truth RGB images (optional)
        
        Returns:
            dict: Evaluation metrics + predictions
        """
        if self.model_type == 'autoencoder':
            processed_data = preprocess(validation_data)
            predictions = self.model.predict(processed_data)

            metrics = {}
            if ground_truth is not None:
                metrics['psnr'] = np.mean([psnr(gt, pred) for gt, pred in zip(ground_truth, predictions)])
                metrics['ssim'] = np.mean([ssim(gt, pred, channel_axis=-1, data_range=1.0)
                                           for gt, pred in zip(ground_truth, predictions)])

            if hasattr(self.model, 'evaluate'):
                evaluation_metrics = self.model.evaluate(processed_data)
                metrics['loss'] = evaluation_metrics[0]
                if len(evaluation_metrics) > 1:
                    metrics['accuracy'] = evaluation_metrics[1]

            self.result.save_evaluation(metrics)
            return {'metrics': metrics, 'predictions': predictions}

        # --- GAN Evaluation ---
        if isinstance(validation_data, np.ndarray):
            if len(validation_data.shape) == 2:
                L = validation_data[np.newaxis, np.newaxis, :, :] / 50.0 - 1.0
            elif len(validation_data.shape) == 3 and validation_data.shape[2] == 1:
                L = validation_data.transpose(0, 3, 1, 2) / 50.0 - 1.0
            elif len(validation_data.shape) == 3:
                lab = rgb2lab(validation_data).astype('float32')
                L = lab[:, :, 0][np.newaxis, np.newaxis, :, :] / 50.0 - 1.0
            elif len(validation_data.shape) == 4 and validation_data.shape[3] == 3:
                L_list, ab_list = [], []
                for img in validation_data:
                    lab = rgb2lab(img).astype('float32')
                    L_list.append(lab[:, :, 0])
                    if ground_truth is None:
                        ab_list.append(lab[:, :, 1:3])
                L = np.stack(L_list)[:, np.newaxis, :, :] / 50.0 - 1.0
                if ground_truth is None:
                    ab_truth = np.stack(ab_list) / 110.0
            else:
                raise ValueError(f"Unsupported input shape: {validation_data.shape}")
            L_tensor = torch.from_numpy(L).float().to(self.device)
        else:
            raise ValueError("Validation data must be a NumPy array")

        with torch.no_grad():
            ab_pred = self.model(L_tensor)
        ab_pred_np = ab_pred.cpu().numpy()
        L_np = (L_tensor.cpu().numpy() + 1.0) * 50.0

        rgb_images = []
        for i in range(L_np.shape[0]):
            L_img = L_np[i, 0, :, :]
            ab_img = ab_pred_np[i] * 110.0
            lab_img = np.concatenate([L_img[:, :, np.newaxis], ab_img.transpose(1, 2, 0)], axis=2)
            rgb_img = lab2rgb(lab_img)
            rgb_images.append(rgb_img)
        rgb_images = np.stack(rgb_images)

        metrics = {}
        if ground_truth is not None:
            if len(ground_truth.shape) == 4 and ground_truth.shape[3] == 3:
                gt_rgb = ground_truth
            else:
                raise ValueError("Ground truth must be RGB images in shape (N, H, W, 3)")

            psnr_values = []
            ssim_values = []
            for i in range(len(rgb_images)):
                psnr_val, ssim_val = calculate_metrics(rgb_images[i], gt_rgb[i])
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)

            metrics['psnr'] = np.mean(psnr_values)
            metrics['ssim'] = np.mean(ssim_values)
            metrics['psnr_std'] = np.std(psnr_values)
            metrics['ssim_std'] = np.std(ssim_values)

        self.result.save_evaluation(metrics)
        return {
            'metrics': metrics,
            'predictions': rgb_images
        }
