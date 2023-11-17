import cv2
import torch
import os

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
    depth_map_equalized = cv2.equalizeHist(depth_map)

    # Apply edge detection (optional)
    edges = cv2.Canny(depth_map_equalized, 100, 200)
    return cv2.bitwise_and(depth_map_equalized, depth_map_equalized, mask=edges)

def overlay_depth_map(original_img, depth_map):
    # Resize depth map to match the original image size
    depth_map_resized = cv2.resize(depth_map, (original_img.shape[1], original_img.shape[0]))

    # Enhance the depth map
    enhanced_depth_map = enhance_depth_map(depth_map_resized)

    # Normalize and colorize depth map
    depth_map_normalized = cv2.normalize(enhanced_depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized.astype('uint8'), cv2.COLORMAP_JET)

    # Overlay the depth map on the original image
    alpha = 0.6  # transparency level
    overlay = cv2.addWeighted(original_img, alpha, depth_map_colored, 1 - alpha, 0)
    return overlay

def process_image(image_path, depth_output_path, overlay_output_path):
    # Read and transform the image
    original_img = cv2.imread(image_path)
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    # Predict and create the depth map
    with torch.no_grad():
        prediction = midas(input_batch)

    # Convert prediction to numpy array
    prediction = prediction.squeeze().cpu().numpy()
    depth_map = (prediction * 255).astype('uint8')

    # Save the enhanced depth map
    enhanced_depth_map = enhance_depth_map(depth_map)
    cv2.imwrite(depth_output_path, enhanced_depth_map)

    # Create an overlay of the depth map on the original image
    overlay_img = overlay_depth_map(original_img, enhanced_depth_map)

    # Save the overlay image
    cv2.imwrite(overlay_output_path, overlay_img)

# Directory containing test images
test_image_dir = 'test/'  # Update this path
output_dir = 'output1/'    # Update this path

# Process each test image
for image_name in os.listdir(test_image_dir):
    if not image_name.endswith('.png'):  # Filter to process only PNG images
        continue

    image_path = os.path.join(test_image_dir, image_name)
    depth_image_path = os.path.join(output_dir, image_name.replace('.png', '_depth_0000.png'))
    overlay_image_path = os.path.join(output_dir, image_name.replace('.png', '_overlay.png'))

    process_image(image_path, depth_image_path, overlay_image_path)

print("Depth maps and overlay images generated in directory: {}".format(output_dir))
