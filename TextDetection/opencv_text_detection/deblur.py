import numpy as np
import cv2, os
from skimage import color, restoration
from scipy.signal import convolve2d as conv2

path = "../All_in_one/"

l = os.listdir(path)
apply = lambda x: path+x
orig_imgs = list(map(apply, l))

psf = np.ones((5, 5))/25
path = "../UnblurredImages/"


for img in orig_imgs:
    # img = "/home/aayush/Documents/RE/opencv-text-detection/All_in_one/Data_phase_1_2_Amit_image_497.jpg"
    filename = os.path.basename(img)
    new_path = path+filename

    image = cv2.imread(img)
    image = color.rgb2gray(image)
    image = conv2(image, psf, "same")
    new_image = restoration.richardson_lucy(image, psf, 300)
    new_image = new_image.astype('float32') * 255.
    # cv2.imshow(filename, new_image)
    # cv2.waitKey(0)
    # break
    cv2.imwrite(new_path, new_image)