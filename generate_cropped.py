import cv2
import os
from tqdm import tqdm

DATA_DIR = "data/MetalDAM/images"
OUTPUT_DIR = "data/MetalDAM/cropped_images"

def crop_images(input_dir, output_dir):
    '''
    Crops the SEM images from the bottom of micrographs
    '''
    # check if output dir exists. if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # crop coordinates (pre-defined)
    # all SEM images are 1024 x 768
    # all labeled images are 1024 x 703
    xi, xf, yi, yf = 0, 1024, 0, 703
        # Loop through each file in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', 'tiff')): 
            # Read the image
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)

            # Perform the crop
            cropped_img = img[yi:yf, xi:xf]

            # Save the cropped image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cropped_img)

def main():
    # crop the images
    crop_images(DATA_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()