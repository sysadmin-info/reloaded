import os
import cv2
import requests
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env
load_dotenv()

# Retrieve the image URL from environment
IMAGE_URL = os.getenv('NA_SMARTFONA_URL')
if not IMAGE_URL:
    raise ValueError("Environment variable 'NA_SMARTFONA_URL' is not set.")

# Determine the directory of this script
BASE_DIR = Path(__file__).resolve().parent

# Define output file name in the same directory as this script
HORIZONTAL_OUTPUT_NAME = 'na_smartfona_stretched_horizontal.png'

# Download the image
try:
    # Append common image extension if missing
    if not IMAGE_URL.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        for ext in ['.png', '.jpg', '.jpeg']:
            test_url = IMAGE_URL + ext
            try:
                head = requests.head(test_url)
                if head.status_code == 200:
                    IMAGE_URL = test_url
                    break
            except requests.RequestException:
                continue
    response = requests.get(IMAGE_URL)
    response.raise_for_status()
except requests.RequestException as e:
    raise RuntimeError(f"Failed to download image from URL: {IMAGE_URL}\n{e}")

# Decode the downloaded content to an OpenCV image
image_array = cv2.imdecode(
    np.frombuffer(response.content, dtype=np.uint8),
    cv2.IMREAD_UNCHANGED
)
if image_array is None:
    raise RuntimeError(f"Failed to decode image from URL: {IMAGE_URL}")

# Get original dimensions
original_height, original_width = image_array.shape[:2]

# Stretch horizontally by 500%
horizontal_factor = 5.0
horizontal_stretched = cv2.resize(
    image_array,
    (int(original_width * horizontal_factor), original_height)
)

# Save the horizontally stretched image
horizontal_output_path = BASE_DIR / HORIZONTAL_OUTPUT_NAME
cv2.imwrite(str(horizontal_output_path), horizontal_stretched)

# Function to retrieve the output path
def get_output_path():
    return str(horizontal_output_path)

if __name__ == '__main__':
    print(f"Horizontal image saved to: {get_output_path()}")
