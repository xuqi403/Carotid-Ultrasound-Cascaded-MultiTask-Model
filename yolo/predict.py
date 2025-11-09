import os
from tqdm import tqdm
from PIL import Image
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    # Mode: Set to "dir_predict" for batch processing and cropping ROIs
    mode = "dir_predict"
    # Directory paths for input images and saving cropped ROIs
    dir_origin_path = ""  # Input directory with ultrasound images
    dir_save_path = ""  # Output directory for cropped ROIs

    if mode == "dir_predict":
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                # Detect and crop ROI; assumes yolo.detect_image handles cropping and saving
                r_image = yolo.detect_image(image, crop=True, crop_save_path=dir_save_path)
                if r_image is None:
                    continue
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                # Save the full image with bounding box (optional; cropped ROI saved separately in yolo.py)
                r_image.save(os.path.join(dir_save_path, img_name), quality=100, subsampling=0)

    else:
        raise AssertionError("Please specify the correct mode: 'dir_predict'.")
