from logging import getLogger

import cv2
import numpy as np

log = getLogger(__name__)


def get_cam_mat(intr_mat, dstr_co, shape):
    """Determine the new Camera Matrix."""
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
        [-width_half, height_half],
        [width_half, height_half],
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
    """Determine the translation matrix of two transformed images.

    When two images have been transformed by an homography, it's possible
    that they are not aligned with the displayed area anymore. So they need to
    be translated.
    """
    # get origina width and height of images
    w_l, h_l = shape_l
    w_r, h_r = shape_r
    log.debug('(h_l,w_l) = {}'.format(shape_l))
    corners_l = np.float32([
        [0, 0],
        [0, w_l],
        [h_l, w_l],
        [h_l, 0]
    ]).reshape(-1, 1, 2)
    corners_r = np.float32([
        [0, 0],
        [0, w_r],
        [h_r, w_r],
        [h_r, 0]
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


def find_rect(quadri_left, quadri_right):
    """Estimate rectangle depend on two quadriliterals.

    The following steps will determine the dimension, of the rectangle/s to
    which the quadrilaterals will be mapped to.
    """
    """
    Rename corners for better orientation.
    ul_l----um_l / um_r----ul_r
     |         |    |        |
     |  left   |    | right  |
     |         |    |        |
    dl_l----dm_l / dm_r----dl_r
    """
    ul_l = quadri_left[0]
    um_l = quadri_left[1]
    dm_l = quadri_left[2]
    dl_l = quadri_left[3]

    um_r = quadri_right[0]
    ul_r = quadri_right[1]
    dl_r = quadri_right[2]
    dm_r = quadri_right[3]

    # get the euclidean distances between the corners of the quadrilaterals
    u_l = np.linalg.norm(ul_l - um_l)
    d_l = np.linalg.norm(dl_l - dm_l)
    l_l = np.linalg.norm(ul_l - dl_l)
    r_l = np.linalg.norm(um_l - dm_l)
    u_r = np.linalg.norm(ul_r - um_r)
    d_r = np.linalg.norm(dl_r - dm_r)
    l_r = np.linalg.norm(ul_r - dl_r)
    r_r = np.linalg.norm(um_r - dm_r)

    hor_l = max(u_l, d_l)
    hor_r = max(u_r, d_r)
    vert = max(l_l, r_l, l_r, r_r)

    """
    Declare the dimension of the destination rectangle.

    rect_dest:
    0 ----  1 ----- 2
    |       |       |
    | left  | right |
    |       |       |
    5 ----  4 ----- 3
    """

    rect_dest = np.zeros((6, 2), np.float32)
    rect_dest[0] = 0, 0
    rect_dest[1] = hor_l, 0
    rect_dest[2] = hor_l + hor_r, 0
    rect_dest[3] = hor_l + hor_r, vert
    rect_dest[4] = hor_l, vert
    rect_dest[5] = 0, vert
    return rect_dest, hor_l


def find_homographys(quadri_left, quadri_right, rect_dest):
    """Determine the homography between (quadri_left, quadri_right) & rect_dest.

    The function will map the the quadrilaterals quadri_left and quadri_right
    to the rectangle rect_dest and return the homography.
    """
    rect_left = np.array(
        [rect_dest[0], rect_dest[1], rect_dest[4], rect_dest[5]])
    rect_right = np.array(
        [rect_dest[1], rect_dest[2], rect_dest[3], rect_dest[4]])
    left_h, m = cv2.findHomography(quadri_left, rect_left)
    right_h, m = cv2.findHomography(quadri_right, rect_right)
    return left_h, right_h


def sort_pts(pts):
    r"""Sort points as convex quadrilateral.

    Sort points in clockwise order, so that they form a convex quadrilateral.

    pts:                sort_pts:
         x   x                      A---B
                      --->         /     \
       x       x                  D-------C

    """
    sort_pts = np.zeros((len(pts), 2), np.float32)
    for i in range(len(pts)):
        sort_pts[i] = pts[argsort_pts(pts)[i]]
    return sort_pts


def argsort_pts(points):
    r"""Sort points as convex quadrilateral.

    Returns the indices that will sort the points in clockwise order,
    so that they form a convex quadrilateral.

    points:                quadri:
         x   x                      A---B
                      --->         /     \
       x       x                  D-------C

    """
    assert (len(points) == 4)

    # calculate the barycentre / centre of gravity
    barycentre = np.zeros((1, 2), np.float32)
    barycentre = points.sum(axis=0) / 4

    # var for saving the points in realtion to the barycentre
    bary_vectors = np.zeros((4, 2), np.float32)

    # var for saving the closest point of the origin
    A = None
    min_dist = None

    for i, point in enumerate(points):

        # determine the distance to the origin
        cur_dist_origin = np.linalg.norm(point)

        # save the closest point of the orgin
        if A is None or cur_dist_origin < min_dist:
            min_dist = cur_dist_origin
            A = i

        # determine point in relation to the barycentre
        bary_vectors[i] = point - barycentre

    angles = np.zeros(4, np.float32)
    # determine the angles of the different points in relation to the line
    # between cloest point of origin (A) and barycentre
    for i, bary_vector in enumerate(bary_vectors):
        if i != A:
            cur_angle = np.arctan2(
                (np.linalg.det((bary_vectors[A], bary_vector))), np.dot(
                    bary_vectors[A], bary_vector))
            if cur_angle < 0:
                cur_angle = 2 * np.pi + cur_angle
            angles[i] = cur_angle
    return np.argsort(angles)
