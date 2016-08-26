from context import composer
import composer.core
import numpy as np
import cv2
from composer import imgtools
import logging.config
# import sys

logging.config.fileConfig('logging_config.ini')

def draw_makers(img, pts, color=(0, 0, 255), marker_types=cv2.MARKER_TILTED_CROSS):
    img_m = np.copy(img)
    pts = pts[0].astype(int)
    for pt in pts:
        cv2.drawMarker(img_m, tuple(pt), color, markerType=marker_types, markerSize=40, thickness=5)
    return img_m

# Load both testing images
img_left_org = cv2.imread(
    'data/20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
img_right_org = cv2.imread(
    'data/20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
nc = composer.core.Composer.create_from_file('composer_params.npz')

# define points which will be drawn (fixed) on the img.
pts_left_org = np.array([[[353, 400], [369, 2703], [3647, 155], [3647, 2737], [
                    1831, 1412], [361, 1522], [3650, 1208], [1750, 172]]]).astype(np.float64)
pts_right_org = np.array([[[428, 80], [429, 1312], [419, 2752], [3729, 99], [
                     3708, 1413], [3683, 2704], [2043, 1780], [2494, 206]]]).astype(np.float64)

pts_left = np.copy(pts_left_org)
pts_right =  np.copy(pts_right_org)



img_left = draw_makers(img_left_org, pts_left, (255, 0, 0), cv2.MARKER_CROSS)
img_right = draw_makers(img_right_org, pts_right, (255, 0, 0), cv2.MARKER_CROSS)


cv2.imwrite('1_0.jpg', img_left)
cv2.imwrite('1_1.jpg', img_right)

# Rectifies both images
img_left, img_right = nc._rectify_images(img_left, img_right)
# pts_left = nc.map_coordinates(pts_left)
pts_left = imgtools.rectify_pts(
    pts_left, nc.intrinsic_matrix, nc.distortion_coeff)
pts_right = imgtools.rectify_pts(
    pts_right, nc.intrinsic_matrix, nc.distortion_coeff)
# pts_right = nc.map_coordinates(pts_right)

img_left_d = draw_makers(img_left, pts_left)
img_right_d = draw_makers(img_right, pts_right)
cv2.imwrite('2_0.jpg', img_left_d)
cv2.imwrite('2_1.jpg', img_right_d)
cv2.imwrite('2_p0.jpg', img_left)
cv2.imwrite('2_p1.jpg', img_right)

# Rotate both images
img_left, img_right = nc._rotate_images(img_left, img_right)
pts_left = cv2.transform(pts_left, nc.left_rot_mat)
pts_right = cv2.transform(pts_right, nc.right_rot_mat)
img_left_d = draw_makers(img_left, pts_left)
img_right_d = draw_makers(img_right, pts_right)
cv2.imwrite('3_0.jpg', img_left_d)
cv2.imwrite('3_1.jpg', img_right_d)
cv2.imwrite('3_p0.jpg', img_left)
cv2.imwrite('3_p1.jpg', img_right)
print(pts_left)
print(nc.left_rot_mat)
# print(nc.map_coordinates(pts_left_org))

result = nc.composePanorama(img_left, img_right)
cv2.imwrite('result_panormat.jpg', result)
pts_left = cv2.perspectiveTransform(pts_left, nc.left_trans)
pts_right = cv2.perspectiveTransform(pts_right, nc.right_trans)
print(pts_left)
print(nc.left_trans)

# pts_left = cv2.perspectiveTransform(pts_left, nc.left_trans)
# pts_right = cv2.perspectiveTransformsform(pts_right, nc.right_trans)
result_m = draw_makers(result, pts_left)
result_m = draw_makers(result_m, pts_right)
cv2.imwrite('result_panormat_m.jpg', result_m)
