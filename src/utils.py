import cv2
import os


def load_image(image_path, mode=cv2.IMREAD_COLOR):
    return cv2.imread(image_path, mode)


def load_files_in_directory(dir):
    return os.listdir(dir)


def load_filepaths_in_directory(dir):
    return [os.path.join(dir, file) for file in os.listdir(dir)]


def save_image(img, path):
    cv2.imwrite(path, img)