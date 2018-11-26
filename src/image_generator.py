import cv2
import numpy as np
from color_transfer import color_transfer
from rotation import rotate_image_pad
from augmentation import augment_image, get_heavy_augmentator, get_light_augmentator


def rotate_image(img, degress):
    rows, cols, depth = img.shape
    rot_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), degress, 1)
    return cv2.warpAffine(img, rot_matrix, (cols, rows))


def resize_image(img, resolution):
    return cv2.resize(img, resolution, interpolation=cv2.INTER_AREA)


#to be refined
def affine_transform(img, points):
    rows, cols, depth = img.shape
    affine_matrix = cv2.getAffineTransform(points[0], points[1])
    return cv2.warpAffine(img, affine_matrix, (cols, rows))


def get_resize_resolution(logo, background, scale):
    height = int(background.shape[0] * scale)
    r = height / float(logo.shape[0])
    dim = (int(logo.shape[1] * r), height)
    return dim


def get_affine_transformation_points(aff_range):
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + aff_range * np.random.uniform() - aff_range / 2
    pt2 = 20 + aff_range * np.random.uniform() - aff_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    return pts1, pts2


def merge_images(transparent, background):
    if transparent.shape[:2] != background.shape[:2]:
        raise ValueError("Dimensions should match")
    alpha_s = transparent[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        background[:, :, c] = (alpha_s * transparent[:, :, c] +
                               alpha_l * background[:, :, c])
    return background


def crop_image_by_alpha(img):
    alpha_channel = img[:, :, 3]
    x_left, y_top = (0, 0)
    x_right = img.shape[1] - 1
    y_bottom = img.shape[0] - 1
    for i in range(img.shape[0]):
        if np.sum(alpha_channel[i, :]) == 0:
            y_top = i
        else:
            break
    for i in reversed(range(img.shape[0])):
        if np.sum(alpha_channel[i, :]) == 0:
            y_bottom = i
        else:
            break
    for i in range(img.shape[1]):
        if np.sum(alpha_channel[:, i]) == 0:
            x_left = i
        else:
            break
    for i in reversed(range(img.shape[1])):
        if np.sum(alpha_channel[:, i]) == 0:
            x_right = i
        else:
            break
    return img[y_top:y_bottom + 1, x_left:x_right + 1, :]


def augment_brightness_camera_images(image):
    alpha_channel = image[:, :, 3]
    if image.shape[2] == 4:
        image = image[:, :, :3]
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .15 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2]*random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return add_alpha_channel(image1, alpha_channel)


def add_alpha_channel(img, alpha):
    b_channel, g_channel, r_channel = cv2.split(img)
    return cv2.merge((b_channel, g_channel, r_channel, alpha))


def position_logo(resolution, logo):
    padding_y = int(logo.shape[0]*0.5)
    padding_x = int(logo.shape[1]*0.5)
    x = np.random.randint(0, max(resolution[1] - padding_x, 10))
    y = np.random.randint(0, max(resolution[0] - padding_y, 10))
    padded = np.zeros((resolution[0], resolution[1], logo.shape[2]))
    height, width = min(logo.shape[0], resolution[0] - y), min(logo.shape[1], resolution[1] - x)
    padded[y:min(y+logo.shape[0], resolution[0]), x:min(x+logo.shape[1], resolution[1]), :] \
        = logo[:height, :width, :]
    return x, y, height, width, padded


def transform_image(logo, background, rotation=45, scale_ratio=0.5, scale=0.1, flip_prob=0.2):

    if len(background.shape) < 3:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    image_part = np.random.uniform(low=0.1, high=scale_ratio, size=(1,))[0]
    rot = np.random.randint(-rotation, rotation)

    rotated = rotate_image_pad(logo, rot)
    affined = affine_transform(rotated, get_affine_transformation_points(10))
    cropped = crop_image_by_alpha(affined)
    final_scale = image_part*np.random.uniform(low=scale, high=1.0, size=(1,))[0]
    scaled = resize_image(cropped, get_resize_resolution(cropped, background, final_scale))

    if np.random.uniform(low=0.0, high=1.0) < flip_prob:
        scaled = cv2.flip(scaled, 0)
    if np.random.uniform(low=0.0, high=1.0) < flip_prob:
        scaled = cv2.flip(scaled, 1)

    brightened = augment_brightness_camera_images(scaled)
    augmented = augment_image(brightened, get_light_augmentator())
    x, y, height, width, padded = position_logo(background.shape[:2], augmented)
    merged = merge_images(padded, background)
    result = color_transfer(background, merged)
    if np.random.uniform(low=0.0, high=1.0) < 0.5:
        result = augment_image(result, get_heavy_augmentator(0.2))
    return result, x, y, height, width


