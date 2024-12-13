from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm
from controlnet_aux import OpenposeDetector

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(device)


def load_and_preprocess_image(
    image_path, target_size=(512, 512), 
    apply_edge_detection=False, 
):
    # Load the image
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    
    # Get the dimensions of the image
    w, h = image.size
    
    # Calculate the scaling factor
    scale = min(target_size[0] / h, target_size[1] / w)
    
    # Resize the image while maintaining aspect ratio
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new('RGB', target_size, (0, 0, 0))  # Create a black image
    pad_w = (target_size[0] - new_w) // 2
    pad_h = (target_size[1] - new_h) // 2
    new_image.paste(image_resized, (pad_w, pad_h))
    
    # Convert to NumPy array
    if apply_edge_detection:
        # Convert to grayscale for Canny edge detection
        image_array = np.array(new_image)
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 100, 200)
        
        # Expand edges into 3 channels
        edges_3d = np.stack([edges] * 3, axis=-1)
        
        # Flatten the 3D edge-detected image
        edges_flattened = edges_3d.flatten()
        cv2.imwrite("test_edge.png", edges)
        return edges_flattened
        
    else:
        # Flatten the image without edge detection
        image_array = np.array(new_image)
        image_flattened = image_array.flatten()
        return image_flattened

def compute_mutual_information(image1, image2, bins=256):
    # Compute the 2D histogram
    joint_hist, _, _ = np.histogram2d(image1, image2, bins=bins)
    
    # Convert the joint histogram to a probability distribution
    joint_prob = joint_hist / np.sum(joint_hist)
    
    # Compute the marginal probabilities
    p1 = np.sum(joint_prob, axis=1)
    p2 = np.sum(joint_prob, axis=0)
    
    # Compute the mutual information
    mutual_info = 0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mutual_info += joint_prob[i, j] * np.log(joint_prob[i, j] / (p1[i] * p2[j]))
    return mutual_info

# Paths to the directories
raw_dir = '../../data/husky/real/raw'
foreground_dir = '../../data/husky/real/raw_split/foreground'
background_dir = '../../data/husky/real/raw_split/background'

# Iterate over all images in the foreground directory
ret1 = []
f_results_L1_L0 = []
f_results_L2_L0 = []
b_results_L1_L0 = []
b_results_L2_L0 = []
for filename in tqdm(os.listdir(foreground_dir)):
    foreground_path = os.path.join(foreground_dir, filename)
    background_path = os.path.join(background_dir, filename)
    raw_path = os.path.join(raw_dir, filename)
    
    f_image_L1 = load_and_preprocess_image(foreground_path, apply_edge_detection=True)
    f_image_L2 = load_and_preprocess_image(foreground_path)
    b_image_L2 = load_and_preprocess_image(background_path)
    raw_image = load_and_preprocess_image(raw_path)
        
    # Compute mutual information for (L1,L0), w.r.t. foreground object
    mutual_info = compute_mutual_information(f_image_L1, f_image_L2)
    entropy_info = compute_mutual_information(raw_image, raw_image)
    f_results_L1_L0.append(mutual_info / entropy_info)
    
    # Compute mutual information for (L2,L0), w.r.t. foreground object
    mutual_info = compute_mutual_information(f_image_L2, f_image_L2)
    f_results_L2_L0.append(mutual_info / entropy_info)
    
    # Compute mutual information for (L1,L0), w.r.t. background object
    mutual_info = compute_mutual_information(f_image_L1, b_image_L2)
    b_results_L1_L0.append(mutual_info / entropy_info)
    
    # Compute mutual information for (L2,L0), w.r.t. background object
    mutual_info = compute_mutual_information(f_image_L2, b_image_L2)
    b_results_L2_L0.append(mutual_info / entropy_info)

print(f"Sanitizer: husky_L1_L0, Object: foreground, MI: {sum(f_results_L1_L0) / len(f_results_L1_L0)}")
print(f"Sanitizer: husky_L2_L0, Object: foreground, MI: {sum(f_results_L2_L0) / len(f_results_L2_L0)}")
print(f"Sanitizer: husky_L1_L0, Object: background, MI: {sum(b_results_L1_L0) / len(b_results_L1_L0)}")
print(f"Sanitizer: husky_L2_L0, Object: background, MI: {sum(b_results_L2_L0) / len(b_results_L2_L0)}")