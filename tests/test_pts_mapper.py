import logging.config
import os

import cv2
import numpy as np
import composer.core

# import sys

logging.config.fileConfig('logging_config.ini')
print(__file__)


def draw_makers(img, pts, color=(0, 0, 255),
                marker_types=cv2.MARKER_TILTED_CROSS):
    img_m = np.copy(img)
    pts = pts[0].astype(int)
    for pt in pts:
        cv2.drawMarker(img_m, tuple(pt), color, markerType=marker_types,
                       markerSize=40, thickness=5)
    return img_m


# Load both testing images
img_left_org = cv2.imread(
    'data/test_pts_mapper/Input/Cam_0_20161507130847_631282517.jpg')
img_right_org = cv2.imread(
    'data/test_pts_mapper/Input/Cam_1_20161507130847_631282517.jpg')

# define points which will be drawn (fixed) on the img.
pts_left_org = np.array([[[353, 400], [369, 2703], [3647, 155], [3647, 2737], [
    1831, 1412], [361, 1522], [3650, 1208], [1750, 172]]]) \
    .astype(np.float64)
pts_right_org = np.array([[[428, 80], [429, 1312], [419, 2752], [3729, 99], [
    3708, 1413], [3683, 2704], [2043, 1780], [2494, 206]]]) \
    .astype(np.float64)

pts_left = np.copy(pts_left_org)
pts_right = np.copy(pts_right_org)

img_l_m = draw_makers(img_left_org, pts_left, (255, 0, 0), cv2.MARKER_CROSS)
img_r_m = draw_makers(img_right_org, pts_right, (255, 0, 0), cv2.MARKER_CROSS)

if not os.path.exists('./data/test_pts_mapper/Output'):
    os.mkdir('./data/test_pts_mapper/Output')

cv2.imwrite('./data/test_pts_mapper/Output/0_0.jpg', img_l_m)
cv2.imwrite('./data/test_pts_mapper/Output/0_1.jpg', img_r_m)

c = composer.core.Composer()
c.load_arguments('./data/test_pts_mapper/Input/composer_data.npz')

pts_r = c.map_coordinate_right(pts_right_org)

res = c.compose_and_mark(img_left_org, img_right_org, pts_r)
cv2.imwrite('./data/test_pts_mapper/Output/result.jpg', res)
