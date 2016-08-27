import cv2
import numpy as np

def get_cam_mat(intr_mat, dstr_co, shape):
    h, w = shape
    newCameraMatrix, __ = cv2.getOptimalNewCameraMatrix(
        intr_mat, dstr_co, (w, h), 1, (w, h), 0)
    return newCameraMatrix

def get_rot_params(angle, shape):
    """Rotate img around center point by given angle.

    The returned img will be large enough to hold the entire
    new img, with a black background.
    """
    # Get img size
    size = (shape[1], shape[0])
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

    return affine_mat, size_new

def get_translation(shape_l, shape_r, homo_mat_l, homo_mat_r):
    # get origina width and height of images
    h_l, w_l = shape_l
    h_r, w_r = shape_r
    log.debug('(h_l,w_l) = {}'.format(shape_l))
    corners_l = np.float32([
        [0,     0],
        [0,     h_l],
        [w_l,   h_l],
        [w_l,   0]
    ]).reshape(-1, 1, 2)
    corners_r = np.float32([
        [0,     0],
        [0,     h_r],
        [w_r,   h_r],
        [w_r,   0]
    ]).reshape(-1, 1, 2)

    # transform the corners of the images, to get the dimension of the
    # transformed images and stitched image
    corners_tr_l = cv2.perspectiveTransform(corners_l, homo_mat_l)
    corners_tr_r = cv2.perspectiveTransform(corners_r, homo_mat_r)
    pts = np.concatenate((corners_tr_l, corners_tr_r), axis=0)

    # measure the max values in x and y direction to get the translation vector
    # so that whole image will be shown
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]

    # define translation matrix
    trans_m = np.array(
        [[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate
    total_size = (xmax - xmin, ymax - ymin)

    return trans_m, total_size
