"""Various tools for manipulating images."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def rectify_img(img, IntrinsicMatrix, dstr_coeff):
    """Take an image and undistort it."""
    # TODO check size, cmp matl
    h, w = img.shape[:2]
    newCameraMatrix_m, validPixRoi_m = cv2.getOptimalNewCameraMatrix(
        IntrinsicMatrix, dstr_coeff, (w, h), 1, (w, h), 0)

    img_rectified = cv2.undistort(
        img, IntrinsicMatrix, dstr_coeff, None, newCameraMatrix_m)
    return img_rectified

def rectify_pts(pts, IntrinsicMatrix, dstr_coeff, image_size=(4000,3000)):
    newCameraMatrix_m, validPixRoi_m = cv2.getOptimalNewCameraMatrix(
        IntrinsicMatrix, dstr_coeff, image_size, 1, image_size, 0)
    return cv2.undistortPoints(pts, IntrinsicMatrix, dstr_coeff, None, newCameraMatrix_m)


def rotate_image(img, angle):
    """Rotate img around center point by given angle.

    The returned img will be large enough to hold the entire
    new img, with a black background.
    """
    # Get img size
    size = (img.shape[1], img.shape[0])
    center = tuple(np.array(size) / 2)
    (width_half, height_half) = center

    # Convert the 3x2 rotation matrix to 3x3
    map_matrix = np.vstack([cv2.getRotationMatrix2D(center, angle, 1.0),
                            [0, 0, 1]])

    # To get just the rotation
    rotation_matrix = np.matrix(map_matrix[:2, :2])

    # Declare the corners of the image in relation to the center
    corners = [
        [-width_half,  height_half],
        [width_half,  height_half],
        [-width_half, -height_half],
        [width_half, -height_half]
    ]

    # get the rotated corners
    corners_rotated = corners * rotation_matrix

    # Find the size of the new img
    x_max = max(corners_rotated[:4, 0])
    x_min = min(corners_rotated[:4, 0])
    y_max = max(corners_rotated[:4, 1])
    y_min = min(corners_rotated[:4, 1])

    # get the new size
    w_rotated = int(abs(x_max - x_min))
    h_rotated = int(abs(y_max - y_min))
    size_new = (w_rotated, h_rotated)

    # matrix to center the rotated image
    translation_matrix = np.matrix([
        [1, 0, int(w_rotated / 2 - width_half)],
        [0, 1, int(h_rotated / 2 - height_half)],
        [0, 0, 1]
    ])

    # get the affine Matrix
    affine_mat = (translation_matrix * map_matrix)[0:2]

    # transformation
    retval = cv2.warpAffine(
        img,
        affine_mat,
        size_new,
        flags=cv2.INTER_LINEAR
    )

    return retval, affine_mat


def add_alpha_channel(img):
    '''add alpha channel to image'''
    # TODO gesamte fkt ersetztbar durch cv2.cvtColor() fkt
    alpha_channel = np.ones((img.shape[:2]), np.uint8)*255
    img = cv2.merge((img, alpha_channel))
    return img
