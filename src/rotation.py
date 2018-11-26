import math
import numpy as np
import cv2


def rotate_image_pad(image, angle):

    diagonal = int(math.ceil(math.sqrt(pow(image.shape[0], 2) + pow(image.shape[1], 2))))
    offset_x = int((diagonal - image.shape[0])/2)
    offset_y = int((diagonal - image.shape[1])/2)
    dst_image = np.zeros((diagonal, diagonal, 4), dtype='uint8')
    image_center = (float(diagonal-1)/2, float(diagonal-1)/2)

    R = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    dst_image[offset_x:(offset_x + image.shape[0]), offset_y:(offset_y + image.shape[1]), :] = image
    dst_image = cv2.warpAffine(dst_image, R, (diagonal, diagonal), flags=cv2.INTER_LINEAR)

    # Calculate the rotated bounding rect
    x0 = offset_x
    x1 = offset_x + image.shape[0]
    x2 = offset_x + image.shape[0]
    x3 = offset_x

    y0 = offset_y
    y1 = offset_y
    y2 = offset_y + image.shape[1]
    y3 = offset_y + image.shape[1]

    corners = np.zeros((3,4))
    corners[0,0] = x0
    corners[0,1] = x1
    corners[0,2] = x2
    corners[0,3] = x3
    corners[1,0] = y0
    corners[1,1] = y1
    corners[1,2] = y2
    corners[1,3] = y3
    corners[2:] = 1

    c = np.dot(R, corners)

    x = int(round(c[0,0]))
    y = int(round(c[1,0]))
    left = x
    right = x
    up = y
    down = y

    for i in range(4):
        x = c[0,i]
        y = c[1,i]
        if (x < left): left = x
        if (x > right): right = x
        if (y < up): up = y
        if (y > down): down = y
    h = int(round(down - up))
    w = int(round(right - left))
    left = int(round(left))
    up = int(round(up))

    cropped = np.zeros((w, h, 4), dtype='uint8')
    cropped[:, :, :] = dst_image[left:(left+w), up:(up+h), :]
    return cropped