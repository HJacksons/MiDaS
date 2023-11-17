import cv2
import torch
import os
import numpy as np
# Initialize MiDaS model
model_type = "DPT_Large"  # or "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.default_transform

def enhance_depth_map(depth_map):
    # Increase contrast using histogram equalization
    equalized = cv2.equalizeHist(depth_map)

    # Apply a sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    return sharpened

def process_image(image_path, output_path):
    # Read and transform the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    # Predict and create the depth map
    with torch.no_grad():
        prediction = midas(input_batch)

    # Convert prediction to numpy array
    prediction = prediction.squeeze().cpu().numpy()
    depth_map = (prediction * 255).astype('uint8')

    # Enhance the depth map
    enhanced_depth_map = enhance_depth_map(depth_map)

    # Save the enhanced depth map
    cv2.imwrite(output_path, enhanced_depth_map)

# Directory containing test images
test_image_dir = 'test/'  # Update this path
output_dir = 'output/'    # Update this path

# Process each test image
for image_name in os.listdir(test_image_dir):
    if not image_name.endswith('.png'):  # Filter to process only PNG images
        continue

    image_path = os.path.join(test_image_dir, image_name)
    depth_image_path = os.path.join(output_dir, image_name.replace('.png', '_depth_0000.png'))

    process_image(image_path, depth_image_path)

print("Enhanced depth maps for test images generated in directory: {}".format(output_dir))
