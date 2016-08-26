from context import composer
from composer.core import Composer
import cv2
import numpy as np
import logging
import sys
# import composer.imgtools

handler = logging.StreamHandler(sys.stdout)
logging.basicConfig
logging.getLogger()#.setLevel(logging.DEBUG)
core_log = logging.getLogger("composer")
core_log.addHandler(handler)
core_log.setLevel(logging.DEBUG)
core_log.propagate = True
pts = np.array([[[1.0, 1.0]]])
print(type(pts))
img_left_org = cv2.imread(
    'data/20160807/Cam_01/Cam_0_20161507130847_631282517.jpg')
img_right_org = cv2.imread(
    'data/20160807/Cam_01/Cam_1_20161507130847_631282517.jpg')
nc = Composer.create_from_file('composer_params.npz')
print(pts)
print(nc.map_coordinates(pts))
print('rect')
left_img, right_img = nc._rectify_images(img_left_org, img_right_org)
print('rot')
left_img, right_img = nc._rotate_images(left_img, right_img)
print('compose')
result = nc.composePanorama(left_img, right_img)
cv2.imwrite("loaded_result.png", result)
print(nc)
