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


# define points which will be drawn (fixed) on the img.
pts_left_org = np.array([[[353, 400], [369, 2703], [3647, 155], [3647, 2737], [
                    1831, 1412], [361, 1522], [3650, 1208], [1750, 172]]]).astype(np.float64)
pts_right_org = np.array([[[428, 80], [429, 1312], [419, 2752], [3729, 99], [
                     3708, 1413], [3683, 2704], [2043, 1780], [2494, 206]]]).astype(np.float64)

pts_left = np.copy(pts_left_org)
pts_right = np.copy(pts_right_org)

img_l_m = draw_makers(img_left_org, pts_left, (255, 0, 0), cv2.MARKER_CROSS)
img_r_m = draw_makers(img_right_org, pts_right, (255, 0, 0), cv2.MARKER_CROSS)

cv2.imwrite('./data/pts_rect/0_0.jpg', img_l_m)
cv2.imwrite('./data/pts_rect/0_1.jpg', img_r_m)

c = composer.core.Composer()
c.load_arguments('./data/composer_data.npz')


'''
rectification
'''
#  images
img_l_re, img_r_re = c.rectify_images(img_l_m, img_r_m)
cv2.imwrite('./data/pts_rect/1_0.jpg', img_l_re)
cv2.imwrite('./data/pts_rect/1_1.jpg', img_r_re)

# points
pts_left_re = c.rectify_points(pts_left)
pts_right_re = c.rectify_points(pts_right)

# marker
img_l_re_m = draw_makers(img_l_re, pts_left_re)
img_r_re_m = draw_makers(img_r_re, pts_right_re)
cv2.imwrite('./data/pts_rect/2_0.jpg', img_l_re_m)
cv2.imwrite('./data/pts_rect/2_1.jpg', img_r_re_m)
'''
Rotation
'''
#  images
img_l_ro = c.rotate_img(img_l_re)
img_r_ro = c.rotate_img_r(img_r_re)
cv2.imwrite('./data/pts_rect/3_0.jpg', img_l_ro)
cv2.imwrite('./data/pts_rect/3_1.jpg', img_r_ro)

# points
pts_left_ro = c.rotate_pts_l(pts_left_re)
pts_right_ro = c.rotate_pts_r(pts_right_re)

print(pts_left_re)
print(pts_left_ro)

# marker
img_l_ro_m = draw_makers(img_l_ro, pts_left_ro)
img_r_ro_m = draw_makers(img_r_ro, pts_right_ro)
cv2.imwrite('./data/pts_rect/4_0.jpg', img_l_ro_m)
cv2.imwrite('./data/pts_rect/4_1.jpg', img_r_ro_m)

'''
Homo
'''
#  images
result = c.couple_pano(img_l_ro, img_r_ro)
cv2.imwrite('./data/pts_rect/5.jpg', result)

# points
pts_left_ho = c.apply_homography_l(pts_left_ro)
pts_right_ho = c.apply_homography_r(pts_right_ro)

# marker
result_m = draw_makers(result, pts_left_ho)
result_m = draw_makers(result_m, pts_right_ho)
cv2.imwrite('./data/pts_rect/6.jpg', result_m)
# cv2.imwrite('./data/pts_rect/4_1.jpg', img_r_ro_m)
