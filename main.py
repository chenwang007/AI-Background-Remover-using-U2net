import os
import glob
import torch
import logging
import numpy as np
import cv2
import argparse
from skimage import io, transform
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)

# Save the output image after processing
def save_output(image_name, pred, d_dir):
    try:
        # Convert prediction tensor to numpy array
        predict = pred.squeeze().cpu().data.numpy()
        im = Image.fromarray((predict * 255).astype(np.uint8)).convert('RGB')

        # Resize output to match original image size
        image = io.imread(image_name)
        imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
        imo.save(os.path.join(d_dir, os.path.basename(image_name)))
        logger.info(f"Saved output to {os.path.join(d_dir, os.path.basename(image_name))}")
    except Exception as e:
        logger.error(f"Error saving output image {image_name}: {e}")

# Remove background using the mask
def remove_background(image, mask):
    _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    result = np.ones_like(image) * 255 
    result = image * binary_mask[:, :, np.newaxis]
    return result

# Load the model based on the selected architecture
def load_model(model_name, model_dir):
    if model_name == 'u2net':
        logger.info("Loading U2NET (173.6 MB)")
        model = U2NET(3, 1)
    elif model_name == 'u2netp':
        logger.info("Loading U2NETP (4.7 MB)")
        model = U2NETP(3, 1)
    else:
        raise ValueError("Model name must be 'u2net' or 'u2netp'.")
    
    try:
        model.load_state_dict(torch.load(model_dir))
        logger.info(f"Model loaded from {model_dir}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_dir}: {e}")
        raise

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

# Prepare DataLoader
def prepare_dataloader(img_name_list, batch_size=1):
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]),
    )
    return DataLoader(test_salobj_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Main function for processing the images
def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="U2NET Background Removal")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output images")
    parser.add_argument('--model_name', type=str, choices=['u2net', 'u2netp'], default='u2net', help="Select the model ('u2net' or 'u2netp')")
    parser.add_argument('--model_dir', type=str, default='./saved_models', help="Path to the model .pth file (default: ./saved_models)")

    args = parser.parse_args()

    # Set up directories and paths from user input
    image_dir = args.input_dir
    prediction_dir = args.output_dir
    model_dir = args.model_dir
    model_name = args.model_name

    # Ensure the output directory exists
    os.makedirs(prediction_dir, exist_ok=True)

    # Get list of image files
    img_name_list = glob.glob(os.path.join(image_dir, '*'))
    if not img_name_list:
        logger.error("No images found in the input directory.")
        return

    # Prepare DataLoader
    test_salobj_dataloader = prepare_dataloader(img_name_list)

    # Load the model
    net = load_model(model_name, model_dir)

    # Inference for each image
    for i_test, data_test in enumerate(test_salobj_dataloader):
        try:
            image_path = img_name_list[i_test]
            logger.info(f"Processing image: {os.path.basename(image_path)}")

            # Get image tensor
            inputs_test = data_test['image'].type(torch.FloatTensor).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Perform inference
            d1, _, _, _, _, _, _ = net(inputs_test)
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)

            # Convert prediction to numpy array
            pred_np = pred.squeeze().cpu().data.numpy()

            # Load original image
            original_image = io.imread(image_path)

            # Resize prediction map to match original image size
            prediction_map_resized = transform.resize(pred_np, original_image.shape[:2], mode='constant')

            # Create mask from prediction map
            mask = (prediction_map_resized > 0.5).astype(np.float32)

            # Remove background from the image
            result_image = remove_background(original_image, mask)

            # Save the result
            save_output(image_path, pred, prediction_dir)

        except Exception as e:
            logger.error(f"Error processing image {img_name_list[i_test]}: {e}")

if __name__ == "__main__":
    main()
