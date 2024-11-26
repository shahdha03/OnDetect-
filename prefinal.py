import cv2
import os
import numpy as np
import subprocess
import pickle

 
def create_output_directory():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def crop_largest_component(image):  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    x, y, w, h, _ = stats[largest_label]
    
    # Ensure the coordinates are within the image boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image

def sharpen_edges(image, intensity=1.0):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Convert the result back to uint8 and adjust the intensity of the edges
    sharpened = np.uint8(np.clip(gray - intensity * laplacian, 0, 255))
    
    return sharpened

def crop_blue_regions(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper threshold for blue color in HSV
    lower_blue = np.array([70, 50, 50])  # Lower threshold for blue color 90
    upper_blue = np.array([160, 255, 255])  # Upper threshold for blue color  130

        # Create a binary mask of blue regions
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours of blue regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a white background image
    white_background = np.full_like(image, (255, 255, 255))

        # Draw filled contours of blue regions on the white background
    cv2.drawContours(white_background, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

        # Use the white background image as a mask to replace blue regions with white in the original image
    cropped_image = cv2.bitwise_or(image, white_background)

    return cropped_image  

def zoom_image(img, zoom_factor=0.2):
    if img is None:
        print("Error: Image is not valid.")
        return img
        
    height, width = img.shape[:2]

        # Calculate the size of the cropped region
    crop_height = int(height * (1 - zoom_factor))
    crop_width = int(width * (1 - zoom_factor))

        # Calculate the top-left corner coordinates for cropping
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2

        # Crop the image
    zoomed_img = img[start_y:start_y + crop_height, start_x:start_x + crop_width]

    return zoomed_img

def resize_image(img, width, height):
    resized_img = cv2.resize(img, (width, height))
    return resized_img

def remove_background_otsu(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=5)

    # Create a mask where the foreground (fingerprint) is white and the background is black
    sure_fg = np.uint8(binary_mask > 0)

    # Combine the sure foreground and background to create the final mask
    mask = cv2.bitwise_or(sure_fg, sure_bg)

    # Apply the mask to the input image
    img_no_bg = cv2.bitwise_and(img, img, mask=mask)

    return img_no_bg


def connect_fingerprint_lines(img):
    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Perform morphological closing to connect fingerprint lines
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return closed_img

def extract_fingerprint(img):

    # Convert the input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to convert grayscale to binary image
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Select the contour with the largest area (assuming it represents the fingerprint)
    max_contour = max(contours, key=cv2.contourArea)

    # Find the convex hull of the contour
    hull = cv2.convexHull(max_contour)

    # Create a mask of zeros with the same dimensions as the input image
    mask = np.zeros_like(gray)

    # Draw the convex hull on the mask
    cv2.drawContours(mask, [hull], -1, (255), thickness=cv2.FILLED)

    # Use the mask to extract the fingerprint from the original image
    fingerprint = cv2.bitwise_and(img, img, mask=mask)

    return fingerprint

def remove_background(imgo):
    height, width = imgo.shape[:2]
    mask = np.zeros(imgo.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (5, 5, width - 30, height - 30)
    cv2.grabCut(imgo, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    img1 = imgo * mask[:, :, np.newaxis]
    background = cv2.absdiff(imgo, img1)
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    final = background + img1
    return final
    
def increase_contrast(img, clipLimit=3, size=(16, 16)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=size)
    out = clahe.apply(img)
    return out
    
def normalize_image(img):
    return np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def smoothing(img):
    return cv2.GaussianBlur(img, (5, 5), 0)
    
def resize_and_crop_image(img, target_width, target_height, crop_offset_left=0, crop_offset_right=0, crop_offset_top=0, crop_offset_bottom=0):
    # Resize the image while maintaining aspect ratio
    height, width = img.shape[:2]
    aspect_ratio = width / height
    if width > height:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    resized_img = cv2.resize(img, (new_width, new_height))

    # Calculate the padding needed to make the image fit the target dimensions
    pad_left = max((target_width - new_width) // 2, 0)
    pad_right = max(target_width - new_width - pad_left, 0)
    pad_top = max((target_height - new_height) // 2, 0)
    pad_bottom = max(target_height - new_height - pad_top, 0)

    # Pad the resized image to match the target dimensions
    padded_img = cv2.copyMakeBorder(resized_img, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))  # assuming white background

    # Calculate the cropping region
    start_x = max(crop_offset_left, 0)
    start_y = max(crop_offset_top, 0)
    end_x = padded_img.shape[1] - max(crop_offset_right, 0)
    end_y = padded_img.shape[0] - max(crop_offset_bottom, 0)

    # Crop the image to the target dimensions
    cropped_img = padded_img[start_y:end_y, start_x:end_x]

    return cropped_img

    
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Unable to read image '{img_path}'")
        return
    print(f"Image shape after reading: {img.shape}")
    # Apply preprocessing steps
    #img = zoom_image(img)
    #img = crop_largest_component(img)
    img = crop_blue_regions(img)
    img = sharpen_edges(img, intensity=5)
      # Adjust width and height as needed
    #img = cv2.convertScaleAbs(img)
    #img = remove_background(img)
   # img = to_gray(img)
    img = normalize_image(img)
    #img = smoothing(img)
    img = increase_contrast(img, clipLimit=4, size=(10, 10))
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 2)
    img = cv2.medianBlur(img, 5)
    kernel = np.ones((3, 3), np.uint8)  # Increase the size of the kernel # change this to view the difference
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    #img = resize_image(img, width=500, height=500)
    img = resize_and_crop_image(img, target_width=500, target_height=500, 
                                     crop_offset_left=10, crop_offset_right=10, 
                                     crop_offset_top=10, crop_offset_bottom=10)
    
    # filename = os.path.basename(img_path)
    # output_path = os.path.join(self.output_folder, filename)
    # cv2.imwrite(output_path, img)
    return img

