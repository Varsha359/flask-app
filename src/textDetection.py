# import Craft class
from craft_text_detector import Craft
import numpy as np
import cv2
import os

def detect_text():
    # set image path and export folder directory
    image = '../data/uploaded.jpeg'  # can be filepath, PIL image or numpy array
    output_dir = '../output/'

    # create a craft instance
    craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

    # apply craft text detection and export detected regions to output directory
    prediction_result = craft.detect_text(image)

    # unload models from ram/gpu
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    print("detected")
    
    image_dir = '../output/uploaded_crops'
    file_list = os.listdir(image_dir)
    # Filter the list to include only image files
    image_files = [os.path.join(image_dir, f)
                    for f in file_list if f.endswith(('.png'))]

    for file_path in image_files:
    # read
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # increase contrast
        pxmin = np.min(img)
        pxmax = np.max(img)
        imgContrast = (pxmax - pxmin) / (img - pxmin) * 255

        # increase line width
        kernel = np.ones((3, 3), np.uint8)
        imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)

        # write
        cv2.imwrite(file_path, imgMorph)
    return "Detected text "

